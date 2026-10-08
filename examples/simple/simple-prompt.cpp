/**
根目录下执行:
  cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-cuda --target llama-simple-prompt
  ./build-cuda/bin/llama-simple-prompt -m model.gguf -p "who are you?"
*/


#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 通用小工具
// ============================================================================

using hr_clock = std::chrono::high_resolution_clock;

static const hr_clock::time_point g_t_start = hr_clock::now();

// 自程序启动以来经过的毫秒数
static double ms_since_start() {
    return std::chrono::duration<double, std::milli>(hr_clock::now() - g_t_start).count();
}

// 从某个时刻到现在的耗时（毫秒）
static double ms_since(const hr_clock::time_point & t) {
    return std::chrono::duration<double, std::milli>(hr_clock::now() - t).count();
}

// 把字节数格式化成人类可读字符串（B / KB / MB / GB）
static std::string human_size(double bytes) {
    const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    int    u = 0;
    double v = bytes;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return std::string(buf);
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"who are you?\"\n", argv[0]);
    LOG("\n");
}

// ============================================================================
// 启动阶段：参数、后端、模型
// ============================================================================

// 调试友好回退：命令行没传 -m / -p 时用内置默认值，方便直接按 F5 调试
static void apply_debug_defaults(common_params & params) {
    if (params.model.path.empty()) {
        params.model.path = "models/qwen2.5-7b-instruct-q3_k_m.gguf";
    }
    if (params.prompt.empty()) {
        params.prompt = "who are you?";
    }

    // 固定生成 token 上限，让每次调试运行长度可控
    params.n_predict = 128;
}

static void print_params(const common_params & params) {
    std::string sampler_chain;
    for (const auto & s : params.sampling.samplers) {
        sampler_chain += common_sampler_type_to_str(s) + ";";
    }
    if (!sampler_chain.empty()) {
        sampler_chain.pop_back(); // 去掉末尾的分号
    }

    LOG("========== 已解析的参数 ==========\n");
    LOG("模型路径        : %s\n", params.model.path.c_str());
    LOG("提示词          : %s\n", params.prompt.c_str());
    LOG("n_ctx           : %d\n", params.n_ctx);
    LOG("n_predict       : %d\n", params.n_predict);
    LOG("n_batch         : %d\n", params.n_batch);
    LOG("n_ubatch        : %d\n", params.n_ubatch);
    LOG("n_gpu_layers    : %d\n", params.n_gpu_layers);
    LOG("n_threads       : %d\n", params.cpuparams.n_threads);
    LOG("n_threads_batch : %d\n", params.cpuparams_batch.n_threads);
    LOG("numa            : %d\n", (int) params.numa);
    LOG("split_mode      : %d\n", (int) params.split_mode);
    LOG("temp            : %.4f\n", (double) params.sampling.temp);
    LOG("top_k           : %d\n", params.sampling.top_k);
    LOG("top_p           : %.4f\n", (double) params.sampling.top_p);
    LOG("min_p           : %.4f\n", (double) params.sampling.min_p);
    LOG("seed            : %u\n", params.sampling.seed);
    LOG("samplers        : %s\n", sampler_chain.c_str());
    LOG("==================================\n");
}

// 必须写 enum 关键字：同名函数 ggml_backend_dev_type() 遮蔽了这个枚举标签
static const char * dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "iGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "?";
}

// 打印后端与设备，确认到底有没有用上 CUDA。
// 后端是在 llama_backend_init 里注册的，所以只能在它之后调，否则列表是空的。
static void print_backends() {
    LOG("========== 后端与设备 ==========\n");

    // CUDA 设备数量：先拿到 CUDA 后端，再问它底下挂了几块卡
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    LOG("CUDA 后端已注册 : %s\n", cuda_reg ? "是" : "否（本次构建不含 CUDA）");
    LOG("CUDA 设备数量   : %zu\n", cuda_reg ? ggml_backend_reg_dev_count(cuda_reg) : (size_t) 0);
    LOG("支持 GPU offload: %s\n", llama_supports_gpu_offload() ? "是" : "否");

    LOG("已注册后端 (%zu) :\n", ggml_backend_reg_count());
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        LOG("  - %-8s 设备数 %zu\n", ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
    }

    // 逐块设备打印详情，同时累计 GPU 显存
    size_t n_gpu          = 0;
    size_t vram_total_all = 0;
    size_t vram_free_all  = 0;

    LOG("可用设备 (%zu) :\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        // get_props 一次拿全名称/描述/显存/PCI id/能力位，省得分别调
        // 必须清零：CPU 后端不填 device_id，不清零就会读到野指针
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(ggml_backend_dev_get(i), &props);

        LOG("  [%zu] %-8s %-5s %s\n", i, props.name, dev_type_str(props.type), props.description);
        LOG("       显存      : 共 %s，空闲 %s\n",
            human_size((double) props.memory_total).c_str(),
            human_size((double) props.memory_free).c_str());
        LOG("       device_id : %s（async %d / host_buffer %d / mmap %d）\n",
            props.device_id ? props.device_id : "未知",
            (int) props.caps.async, (int) props.caps.host_buffer, (int) props.caps.mmap_support);

        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU || props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ++n_gpu;
            vram_total_all += props.memory_total;
            vram_free_all  += props.memory_free;
        }
    }

    LOG("---------------------------------\n");
    LOG("GPU 设备总数    : %zu\n", n_gpu);
    LOG("显存合计        : 共 %s，空闲 %s\n",
        human_size((double) vram_total_all).c_str(), human_size((double) vram_free_all).c_str());
    LOG("================================\n");
}

// 第一块独显，没有就退化到集显
static ggml_backend_dev_t find_gpu_device() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

// 加载模型 + 创建上下文 + 构建采样器，这是启动阶段最耗时的一步。
// 内部依次完成：
//   1) 从磁盘读取模型文件并把权重上传到 RAM/VRAM，
//   2) 创建 llama_context（分配 KV cache），
//   3) 根据 params.sampling.samplers 构建采样器链。
static common_init_result_ptr load_model(common_params & params) {
    ggml_backend_dev_t gpu_dev = find_gpu_device();

    size_t vram_free_before = 0;
    if (gpu_dev) {
        size_t vram_total;
        ggml_backend_dev_memory(gpu_dev, &vram_free_before, &vram_total);
    }

    const auto t_load = hr_clock::now();
    LOG("开始：加载模型 + 创建上下文 + 采样器 (common_init_from_params)，距启动 %.1f ms [计时]\n", ms_since_start());
    auto llama_init = common_init_from_params(params);
    LOG("结束：加载模型 + 创建上下文 + 采样器，本段耗时 %.1f ms（距启动 %.1f ms） [计时]\n",
        ms_since(t_load), ms_since_start());

    // 显存空闲量掉下去了，说明权重/KV cache 真的卸载到了 GPU；基本没变就是跑在 CPU 上
    if (gpu_dev) {
        size_t vram_free_after, vram_total;
        ggml_backend_dev_memory(gpu_dev, &vram_free_after, &vram_total);
        LOG("加载后 %s 显存占用增量 : %s（空闲 %zu -> %zu MiB）\n",
            ggml_backend_dev_name(gpu_dev),
            human_size((double) vram_free_before - (double) vram_free_after).c_str(),
            vram_free_before / 1024 / 1024, vram_free_after / 1024 / 1024);
    } else {
        LOG("未发现 GPU 设备，本次推理在 CPU 上运行\n");
    }

    return llama_init;
}

// llama.cpp 的内存主要分三块：
//   1) 模型权重：加载到 RAM（或 mmap 映射），GPU 卸载时部分进 VRAM；
//   2) KV cache（上下文缓存）：随 n_ctx 线性增长，往往比权重还大；
//   3) 计算/临时 buffer：推理时的 scratch、计算图等。
static void print_memory_usage(const llama_model * model, llama_context * ctx) {
    const uint64_t model_size  = llama_model_size(model);          // 模型权重总字节数
    const size_t   state_size  = llama_state_get_size(ctx);        // 序列化整个上下文所需的字节数
    const int32_t  n_ctx       = llama_n_ctx(ctx);                 // 实际上下文长度
    const int32_t  n_layer     = llama_model_n_layer(model);       // 层数
    const int32_t  n_embd      = llama_model_n_embd(model);        // 隐藏维度
    const int32_t  n_ctx_train = llama_model_n_ctx_train(model);   // 模型训练时的上下文长度

    // state 里不含模型权重，所以它近似等于 KV cache + 少量运行时状态
    const double kv_cache_size = (double) state_size;

    LOG("========== 内存占用统计 ==========\n");
    LOG("模型权重 (model)       : %s  (%llu bytes)\n",
        human_size((double) model_size).c_str(), (unsigned long long) model_size);
    LOG("上下文/KV cache (state): %s  (%llu bytes)\n",
        human_size(kv_cache_size).c_str(), (unsigned long long) state_size);
    LOG("  ├─ n_ctx            : %d\n", n_ctx);
    LOG("  ├─ n_layer          : %d\n", n_layer);
    LOG("  ├─ n_embd           : %d\n", n_embd);
    LOG("  └─ 训练上下文 n_ctx_train: %d\n", n_ctx_train);
    LOG("估算每 token 的 KV 占用 : %s\n",
        human_size(kv_cache_size / std::max(1, n_ctx)).c_str());
    LOG("理论 KV 公式校验        : %.2f MB ≈ 2(K,V) * n_layer * n_embd * n_ctx * 类型字节\n",
        (double)(2LL * n_layer * n_embd * n_ctx * 2) / (1024.0 * 1024.0)); // 以 F16(2B) 估算
    LOG("---------------------------------\n");
    LOG("合计 (权重 + 上下文)    : %s\n",
        human_size((double) model_size + kv_cache_size).c_str());
    LOG("==================================\n");
}

// ============================================================================
// 对话阶段
// ============================================================================

// 生成一轮回复。
// 喂进去的是「本轮新增片段」而不是整段历史：历史前缀已经在 KV cache 里，
// 重复喂一遍既慢又会把上下文撑爆。
static std::string generate(llama_context * ctx, common_sampler * smpl, int n_predict, const std::string & delta) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_memory_t      mem   = llama_get_memory(ctx);

    // 只有第一轮要分词器补 BOS；模板引擎已经把文本里的 BOS 剥掉了（见 common/chat.cpp）
    const bool is_first = llama_memory_seq_pos_max(mem, 0) == -1;

    std::vector<llama_token> tokens = common_tokenize(ctx, delta, is_first, true);

    const int n_ctx      = (int) llama_n_ctx(ctx);
    const int n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
    const int n_reserve  = n_predict > 0 ? n_predict : 0;

    if (n_ctx_used + (int) tokens.size() + n_reserve > n_ctx) {
        LOG_ERR("上下文不够：已用 %d + 本轮 %d + 预留 %d > n_ctx %d，请 /clear 或加大 -c\n",
                n_ctx_used, (int) tokens.size(), n_reserve, n_ctx);
        return std::string();
    }

    std::string response;
    llama_token id    = 0;
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    int        n_gen = 0;
    const auto t_gen = hr_clock::now();

    for (int i = 0; n_predict < 0 || i < n_predict; i++) {
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: 解码失败\n", __func__);
            break;
        }

        id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        const std::string piece = common_token_to_piece(ctx, id);
        LOG("%s", piece.c_str());
        response += piece;
        ++n_gen;

        // id 必须活到下一次 llama_decode 之后，所以放在循环作用域之外
        batch = llama_batch_get_one(&id, 1);
    }

    LOG("\n");
    LOG("本轮：输入 %d tokens（其中历史已缓存 %d），生成 %d tokens，耗时 %.1f ms [计时]\n",
        (int) tokens.size(), n_ctx_used, n_gen, ms_since(t_gen));

    return response;
}

// 原样打印模板文本。里面的 <|im_start|> 之类是普通字符，直接输出即可看清边界
static void print_prompt(const char * title, const std::string & text) {
    LOG("---------- %s（%zu 字节）----------\n", title, text.size());
    LOG("%s\n", text.c_str());
    LOG("----------------------------------------------------\n");
}

enum class cmd_result {
    not_command, // 普通输入，交给模型
    handled,     // 已处理，继续下一轮
    quit,
};

static cmd_result handle_command(const std::string           &  line,
                                 std::vector<common_chat_msg> & history,
                                 std::string                  & full_prompt,
                                 llama_context                * ctx,
                                 common_sampler               * smpl) {
    if (line == "/quit" || line == "/exit") {
        return cmd_result::quit;
    }

    if (line == "/clear") {
        history.clear();
        full_prompt.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        common_sampler_reset(smpl);
        LOG("已清空对话历史和 KV cache\n");
        return cmd_result::handled;
    }

    if (line == "/prompt") {
        print_prompt("累计 prompt：模型看到的全部上下文", full_prompt);
        return cmd_result::handled;
    }

    if (line == "/history") {
        LOG("---------- 对话历史（%zu 条）----------\n", history.size());
        for (size_t i = 0; i < history.size(); ++i) {
            LOG("[%zu] %-9s %s\n", i, history[i].role.c_str(), history[i].content.c_str());
        }
        LOG("--------------------------------------\n");
        return cmd_result::handled;
    }

    return cmd_result::not_command;
}

static void chat_loop(const common_params & params,
                      const llama_model   * model,
                      llama_context       * ctx,
                      common_sampler      * smpl) {
    // 多轮对话必须套聊天模板，否则模型只会把输入当纯文本续写
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);

    // 本次启动后的全部对话，user / assistant 交替追加，永不重排
    std::vector<common_chat_msg> history;

    // 模型实际看到的完整上下文：历次喂入的模板片段 + 模型自己生成的文本
    std::string full_prompt;

    LOG("========== 进入对话（/quit 退出，/clear 清空，/history 历史，/prompt 完整上下文）==========\n");

    // -p 传了就当作第一轮输入，省得每次手敲
    std::string pending = params.prompt;

    while (true) {
        std::string line;

        if (!pending.empty()) {
            line = pending;
            pending.clear();
            LOG("> %s\n", line.c_str());
        } else {
            LOG("> ");
            if (!std::getline(std::cin, line)) {
                break; // Ctrl+D
            }
        }

        if (line.empty()) {
            continue;
        }

        const cmd_result cmd = handle_command(line, history, full_prompt, ctx, smpl);
        if (cmd == cmd_result::quit) {
            break;
        }
        if (cmd == cmd_result::handled) {
            continue;
        }

        common_chat_msg user_msg;
        user_msg.role    = "user";
        user_msg.content = line;

        // 关键一步：拿「历史 + 本轮」相对「历史」的差集，也就是本轮真正新增的那段模板文本
        const std::string delta = common_chat_format_single(tmpls.get(), history, user_msg, true, params.use_jinja);
        history.push_back(user_msg);

        full_prompt += delta;

        print_prompt("本轮新增：真正送进 llama_decode 的部分", delta);
        print_prompt("累计 prompt：模型看到的全部上下文", full_prompt);

        common_chat_msg assistant_msg;
        assistant_msg.role    = "assistant";
        assistant_msg.content = generate(ctx, smpl, params.n_predict, delta);
        history.push_back(assistant_msg);

        // 模型自己生成的 token 也进了 KV cache，是下一轮上下文的一部分
        full_prompt += assistant_msg.content;
    }

    LOG("\n对话结束，共记录 %zu 条消息\n", history.size());
}

// ============================================================================

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // 公共参数解析：读取 -m / -p 等，同时把库的所有默认值（n_ctx、采样器链、设备列表）填好
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    apply_debug_defaults(params);
    print_params(params);

    common_init();

    // === 步骤 1：初始化计算后端（CPU 指令集探测、线程池、BLAS/CUDA/Metal 注册）===
    const auto t_backend = hr_clock::now();
    LOG("开始：初始化计算后端 (llama_backend_init / llama_numa_init)，距启动 %.1f ms [计时]\n", ms_since_start());
    llama_backend_init();
    llama_numa_init(params.numa);
    LOG("结束：初始化计算后端，本段耗时 %.1f ms（距启动 %.1f ms） [计时]\n",
        ms_since(t_backend), ms_since_start());

    print_backends();

    // === 步骤 2：加载模型 ===
    auto llama_init = load_model(params);

    llama_model    * model = llama_init->model();
    llama_context  * ctx   = llama_init->context();
    common_sampler * smpl  = llama_init->sampler(0);

    if (!model || !ctx || !smpl) {
        LOG_ERR("%s: 初始化失败\n", __func__);
        return 1;
    }

    print_memory_usage(model, ctx);

    // === 步骤 3：交互式多轮对话 ===
    chat_loop(params, model, ctx, smpl);

    common_perf_print(ctx, smpl);
    LOG("程序总运行时间：%.1f ms [计时]\n", ms_since_start());

    llama_backend_free();

    return 0;
}
