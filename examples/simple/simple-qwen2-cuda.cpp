/**
 * simple-qwen2-cuda.cpp
 *
 * 本文件做什么
 * ------------
 * 用已有接口做分词, 后面的推理不走 llama_decode / llama_context.
 * 自己读 GGUF, 把权重拷到 GPU, 按 Qwen2 结构搭 ggml 计算图,
 * 然后 ggml_backend_graph_compute 去发射 CUDA kernel.
 *
 * 调用链 (对应下面 main 的几个阶段)
 *   文本
 *     --llama_tokenize--> token id            (CPU, 词表在 GGUF 的 tokenizer.*)
 *     --get_rows-------> 隐状态 [n_embd]
 *     --28 x block-----> Attention + FFN      (GPU, 权重是 Q4_K/Q6_K)
 *     --lm_head--------> logits [n_vocab]
 *     --argmax---------> 下一个 token          (CPU)
 *     --token_to_piece-> 打印到屏幕
 *
 * 不用: llama_context, llama_decode, llama_batch, llama_sampler, common_*
 * 用:   llama 分词, gguf 读元数据, ggml 算子, ggml CUDA backend
 *
 * 为什么不手写 CUDA kernel
 *   模型是 Q4_K, 矩阵乘要先反量化. ggml CUDA 已经有这些 kernel.
 *   ggml_mul_mat 在 GPU backend 上跑, 内部就是 cuBLAS / 量化 matmul kernel.
 *
 * ggml 两个约定, 读图的时候要对上:
 *   1) 张量形状 [ne0, ne1, ne2], ne0 是最密的一维 (类似列主序的"行宽").
 *   2) C = ggml_mul_mat(A, B): A 是 [K, M], B 是 [K, N], C 是 [M, N]
 *      数学上相当于 C = A^T * B. 所以 GGUF 里 Wq [n_embd, n_embd] 直接丢进去就对.
 *
 * 根目录下执行:
 *   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
 *   cmake --build build-cuda --target simple-qwen2-cuda
 *   ./build-cuda/bin/simple-qwen2-cuda -m model.gguf -p "who are you?" -c 512 -n 64
 */

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    define FSEEK _fseeki64
#else
#    define FSEEK fseeko
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 出错就打印并退出. 纯 CPU, 不涉及资源.
static void die(const char * msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

// 日志回调. 干什么: 把 ggml/llama 的日志重定向到 stderr, 丢掉 DEBUG/INFO.
// 为什么: warmup 之类的 INFO 会打到 stdout, 混进模型输出. 这里只放 WARN/ERROR.
// 流程位置: 启动时装一次. CPU 开销: 可忽略.
static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
        return;
    }
    fputs(text, stderr);
    fflush(stderr);
}

// 把字节数格式化成人类可读 (B/KB/MB/GB). 纯 CPU 字符串处理.
static std::string human_size(double bytes) {
    const char * units[] = { "B", "KB", "MB", "GB" };
    int u = 0;
    while (bytes >= 1024.0 && u < 3) {
        bytes /= 1024.0;
        ++u;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", bytes, units[u]);
    return buf;
}

// 查显存. 干什么: 读整卡的 free/total, used = total - free.
// 注意: 这是整张卡 (含别的进程), 不是本程序单独用量. 底层是 cudaMemGetInfo.
// 流程位置: 只用于观测, 不影响推理. CPU 开销: 一次驱动查询, 可忽略.
static void print_vram(ggml_backend_dev_t dev, const char * tag) {
    size_t free_b = 0;
    size_t total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    const size_t used_b = total_b > free_b ? total_b - free_b : 0;
    printf("vram [%s]: used %s / %s  (free %s)\n",
           tag, human_size((double) used_b).c_str(),
           human_size((double) total_b).c_str(),
           human_size((double) free_b).c_str());
}

// 下面三个 kv_* 是读 GGUF 元数据的小工具.
// 干什么: 从 GGUF 头部的 key-value 表里查超参 (层数/维度这些). 只读内存, 不碰 GPU.
// 流程位置: 加载阶段解析模型结构用. CPU 开销: 查表, 可忽略.

// GGUF 里超参键是 "qwen2.embedding_length" 这种 "架构名.字段".
static int64_t kv_find(const gguf_context * gguf, const char * arch, const char * suffix) {
    const std::string key = std::string(arch) + "." + suffix;
    int64_t id = gguf_find_key(gguf, key.c_str());
    if (id >= 0) {
        return id;
    }
    return gguf_find_key(gguf, suffix);
}

static uint32_t kv_u32(const gguf_context * gguf, const char * arch, const char * suffix) {
    const int64_t id = kv_find(gguf, arch, suffix);
    if (id < 0) {
        fprintf(stderr, "missing kv: %s.%s\n", arch, suffix);
        exit(1);
    }
    return gguf_get_val_u32(gguf, id);
}

static float kv_f32_or(const gguf_context * gguf, const char * arch, const char * suffix, float def) {
    const int64_t id = kv_find(gguf, arch, suffix);
    if (id < 0) {
        return def;
    }
    const enum gguf_type t = gguf_get_kv_type(gguf, id);
    if (t == GGUF_TYPE_FLOAT64) {
        return (float) gguf_get_val_f64(gguf, id);
    }
    return gguf_get_val_f32(gguf, id);
}

// 从 GGUF KV 读出来, 用来搭图. 不进 GPU.
struct qwen_hparams {
    uint32_t n_vocab;      // 词表大小, lm_head 的输出宽
    uint32_t n_embd;       // 隐藏维度, 本模型 3584
    uint32_t n_layer;      // Transformer 层数, 本模型 28
    uint32_t n_ff;         // FFN 中间维度, 本模型 18944
    uint32_t n_head;       // Q 头数, 本模型 28
    uint32_t n_head_kv;    // KV 头数, 本模型 4 (GQA: 7 个 Q 头共用 1 个 KV 头)
    uint32_t n_rot;        // 每个头的维度 = n_embd / n_head = 128, 也是 RoPE 作用维
    uint32_t n_ctx_train;  // 训练窗口, 只影响 RoPE, 不决定 KV 分配
    float    rms_eps;
    float    rope_freq_base;
};

// 一层的权重指针. 这些 ggml_tensor 的 data 在 GPU 上.
struct qwen_layer {
    ggml_tensor * attn_norm; // RMSNorm gamma, [n_embd]
    ggml_tensor * wq;        // [n_embd, n_embd]
    ggml_tensor * bq;        // Qwen2 有 Q/K/V bias, Llama 通常没有
    ggml_tensor * wk;        // [n_embd, n_rot * n_head_kv]
    ggml_tensor * bk;
    ggml_tensor * wv;
    ggml_tensor * bv;
    ggml_tensor * wo;        // [n_embd, n_embd]
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_gate;  // SwiGLU 左支 [n_embd, n_ff]
    ggml_tensor * ffn_up;    // SwiGLU 右支
    ggml_tensor * ffn_down;  // [n_ff, n_embd]
    // KV cache 不在 GGUF 里, 运行时另开. 形状 [n_rot*n_head_kv, n_ctx], F32, GPU
    ggml_tensor * k_cache;
    ggml_tensor * v_cache;
};

struct qwen_model {
    qwen_hparams hparams;
    ggml_context * ctx_w;     // 只存权重和 KV 的元数据, no_alloc, 显存在 backend buffer
    ggml_tensor  * tok_embd;  // [n_embd, n_vocab]
    ggml_tensor  * output_norm;
    ggml_tensor  * output;    // lm_head; 有的模型绑到 tok_embd
    std::vector<qwen_layer> layers;
    int n_ctx = 0;            // KV 槽位数, 图的 n_kv 固定成这个, 避免 gallocr 每步 realloc
};

// 建一个权重张量头 (metadata), 不搬数据.
// 干什么: 按名字从 GGUF 目录读出形状/类型, 在 ctx (no_alloc) 里建一个空壳张量.
//   此时 tensor->data 还是空, 真正的字节等后面 ggml_backend_alloc_ctx_tensors 一次性上 GPU.
// 流程位置: 加载阶段, 声明这一层有哪些权重. CPU 开销: 只建结构体, 可忽略.
static ggml_tensor * add_weight(gguf_context * gguf, ggml_context * ctx, const char * name) {
    const int64_t id = gguf_find_tensor(gguf, name);
    if (id < 0) {
        fprintf(stderr, "missing tensor: %s\n", name);
        exit(1);
    }
    const int64_t * ne = gguf_get_tensor_ne(gguf, id);
    const enum ggml_type type = gguf_get_tensor_type(gguf, id);
    int nd = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (ne[d] > 1) {
            nd = d + 1;
        }
    }
    ggml_tensor * t = ggml_new_tensor(ctx, type, nd, ne);
    ggml_set_name(t, name);
    return t;
}

// 注意力核心: out = softmax(Q K^T / sqrt(d)) V.
// 干什么: 只搭"图节点", 不真算. 真正的 GPU kernel 在后面 graph_compute 才跑.
// 流程位置: 每层 attention 的中段, 被 build_qwen_graph 调用.
// 变量: q [d, n_head, n_q]  k/v [d, n_head_kv, n_kv]  mask [n_kv,n_q]  scale=1/sqrt(d).
// GQA 不要 ggml_repeat: repeat 是 i % n_head_kv (交错),
// mul_mat 广播是 i / (n_head/n_head_kv) (连续分组, 和官方 flash attn 一样).
// CPU 开销: 只建节点, 极小; 真正的算力在 GPU.
static ggml_tensor * attn(
        ggml_context * ctx,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * mask,
        float scale) {
    // [d, n_head*, n_seq] -> [d, n_seq, n_head*]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q); // [n_kv, n_q, n_head]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

    v = ggml_cont(ctx, ggml_transpose(ctx, v)); // [n_kv, d, n_head_kv]
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq); // [d, n_q, n_head]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);     // [d, n_head, n_q]
    return ggml_cont_2d(ctx, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2] * kqv->ne[3]);
}

// 搭整张前向计算图 (28 层 Qwen2 block + lm_head).
// 干什么: 只"描述"要算什么 (建 DAG 节点), 不分配显存、不发 kernel.
//   分配在 ggml_gallocr_alloc_graph, 执行在 ggml_backend_graph_compute.
// 流程位置: 加载完权重后建一次, 之后每步复用同一张图.
// 关键设计: 图形状固定成整段 n_ctx KV + mask. 写 cache 用 set_rows, 它的返回值
//   再拿去算 attn -> 写和读之间有数据依赖, CUDA 不会先读到还没写的 KV.
// CPU 开销: 建图是纯 CPU, 但只跑一次, 相对推理可忽略.
static ggml_cgraph * build_qwen_graph(
        ggml_context * ctx,
        qwen_model & model,
        ggml_tensor * tokens,   // I32 [1]
        ggml_tensor * pos,      // I32 [1], RoPE 绝对位置
        ggml_tensor * mask,     // F32 [n_ctx, 1], 0 可见, -inf 屏蔽
        ggml_tensor * kv_idx) { // I64 [1], 本步写入 KV 的行号, 等于 pos
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    const qwen_hparams & hp = model.hparams;
    const int n_ctx = model.n_ctx;
    const int n_tokens = 1;
    const int n_embd_gqa = (int) (hp.n_rot * hp.n_head_kv);
    const float kq_scale = 1.0f / sqrtf((float) hp.n_rot);

    // token id -> 隐状态向量 [n_embd]. 从嵌入表按行取, 不做乘法.
    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, tokens);

    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        qwen_layer & layer = model.layers[il];
        ggml_tensor * residual = cur; // 残差: 这一支旁路, 最后加回来

        // --- attention 前的 RMSNorm ---
        cur = ggml_rms_norm(ctx, cur, hp.rms_eps);
        cur = ggml_mul(ctx, cur, layer.attn_norm);

        // --- 线性投影出 Q/K/V (Qwen2 带 bias) ---
        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, layer.wq, cur), layer.bq);
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, layer.wk, cur), layer.bk);
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, layer.wv, cur), layer.bv);

        // 拆成多头: [n_embd] -> [每头维度 n_rot, 头数, n_tokens]
        Q = ggml_reshape_3d(ctx, Q, hp.n_rot, hp.n_head, n_tokens);
        K = ggml_reshape_3d(ctx, K, hp.n_rot, hp.n_head_kv, n_tokens);
        V = ggml_reshape_3d(ctx, V, hp.n_rot, hp.n_head_kv, n_tokens);

        // RoPE: 用绝对位置 pos 给 Q/K 加旋转位置编码. V 不加.
        Q = ggml_rope_ext(ctx, Q, pos, nullptr, (int) hp.n_rot, GGML_ROPE_TYPE_NEOX,
                          (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, pos, nullptr, (int) hp.n_rot, GGML_ROPE_TYPE_NEOX,
                          (int) hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // 把本步的 K/V 写进 cache 第 kv_idx 行, 再取整段 cache [.., n_ctx] 去算 attention.
        // set_rows 返回写后的 cache, k_all/v_all 依赖它 -> 保证先写后读.
        ggml_tensor * k_row = ggml_reshape_2d(ctx, ggml_cont(ctx, K), n_embd_gqa, n_tokens);
        ggml_tensor * v_row = ggml_reshape_2d(ctx, ggml_cont(ctx, V), n_embd_gqa, n_tokens);
        ggml_tensor * k_all = ggml_reshape_3d(ctx,
                ggml_set_rows(ctx, layer.k_cache, k_row, kv_idx),
                hp.n_rot, hp.n_head_kv, n_ctx);
        ggml_tensor * v_all = ggml_reshape_3d(ctx,
                ggml_set_rows(ctx, layer.v_cache, v_row, kv_idx),
                hp.n_rot, hp.n_head_kv, n_ctx);

        // 注意力 + 输出投影 wo, 再加残差
        ggml_tensor * attn_out = attn(ctx, Q, k_all, v_all, mask, kq_scale);
        cur = ggml_add(ctx, residual, ggml_mul_mat(ctx, layer.wo, attn_out));

        // --- FFN (SwiGLU): norm -> gate/up 升维 -> silu 门控 -> down 降维 -> 残差 ---
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_eps);
        cur = ggml_mul(ctx, cur, layer.ffn_norm);
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, layer.ffn_gate, cur));
        ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, cur);
        cur = ggml_add(ctx, residual, ggml_mul_mat(ctx, layer.ffn_down, ggml_mul(ctx, gate, up)));
    }

    // 最后一层后面的 RMSNorm + lm_head. logits 标 output, 分配器不会覆盖它.
    cur = ggml_rms_norm(ctx, cur, hp.rms_eps);
    cur = ggml_mul(ctx, cur, model.output_norm);
    ggml_tensor * logits = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// 把权重从磁盘搬到 GPU.
// 干什么: 遍历 GGUF 里每个张量, 读它的量化字节到 host 缓冲, 再 tensor_set (内部 cudaMemcpy 到显存).
//   这里不申请显存 (显存已由 alloc_ctx_tensors 开好), 只做"拷贝"这一件事.
// 流程位置: 加载阶段, alloc 之后、推理之前, 跑一次.
// CPU 开销: 主要是磁盘 IO + memcpy. 几个 GB 的模型这一步是启动耗时大头.
static void upload_tensors(gguf_context * gguf, ggml_context * ctx_w, const char * path) {
    FILE * fp = fopen(path, "rb");
    if (!fp) {
        die("open gguf for tensor blob");
    }
    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t n = gguf_get_n_tensors(gguf);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(ctx_w, name);
        if (!t) {
            continue; // 词表之类我们没建张量的, 跳过
        }
        const size_t nb = ggml_nbytes(t);
        buf.resize(nb);
        if (FSEEK(fp, (int64_t) data_off + (int64_t) gguf_get_tensor_offset(gguf, i), SEEK_SET) != 0) {
            die("seek tensor");
        }
        if (fread(buf.data(), 1, nb, fp) != nb) {
            die("read tensor");
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
        if ((i + 1) % 50 == 0 || i + 1 == n) {
            printf("  uploaded %lld / %lld tensors\r", (long long) (i + 1), (long long) n);
            fflush(stdout);
        }
    }
    printf("\n");
    fclose(fp);
}

static void print_usage(const char * argv0) {
    printf("usage: %s -m model.gguf [-p prompt] [-n n_predict] [-c n_ctx]\n", argv0);
}

// 判断用户是否已经手写了对话模板, 避免重复套一层.
static bool already_templated(const std::string & s) {
    return s.find("<|im_start|>") != std::string::npos;
}

// 套聊天模板.
// 干什么: 用 GGUF 自带的 chat template 把 "who are you?" 包成模型认识的对话格式.
//   R1 蒸馏不是 ChatML, 硬套 ChatML 会让它自言自语 (narrate).
// 流程位置: 分词之前. CPU 开销: 字符串拼接, 可忽略. 不碰 GPU.
static std::string apply_chat_prompt(const llama_model * model, const std::string & user) {
    if (already_templated(user)) {
        return user;
    }
    llama_chat_message msg = { "user", user.c_str() };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(user.size() * 4 + 256);
    int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    if (n > (int32_t) buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    }
    if (n <= 0) {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    }
    return std::string(buf.data(), (size_t) n);
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    ggml_log_set(quiet_log, nullptr);
    llama_log_set(quiet_log, nullptr);

    std::string path;
    std::string prompt = "who are you?";
    int n_predict = 64; // 最多新生成多少 token
    int n_ctx = 512;    // KV cache 槽位数. 不要用 n_ctx_train=131072, 显存会爆

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (path.empty()) {
        path = "models/qwen2.5-7b-instruct-q3_k_m.gguf";
        printf("no -m, try %s\n", path.c_str());
    }

    // =====================================================================
    // 1) 分词. 干什么: 文本 -> token id 数组.
    //    这是本文件唯一还在用的 llama 模型接口.
    //    vocab_only=true: 只解析 tokenizer.*, 不把 4GB 权重量进 llama_model (省显存/内存).
    //    资源: 只在 CPU/内存里建词表. CPU 开销: 加载词表几十毫秒, 分词本身极快.
    // =====================================================================
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * vocab_model = llama_model_load_from_file(path.c_str(), mp);
    if (!vocab_model) {
        die("vocab_only load failed");
    }
    const llama_vocab * vocab = llama_model_get_vocab(vocab_model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    prompt = apply_chat_prompt(vocab_model, prompt);

    // 第一次传 NULL, 返回值的绝对值是 token 个数
    const int n_tok = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, add_bos, true);
    if (n_tok <= 0) {
        die("tokenize size");
    }
    std::vector<llama_token> prompt_toks(n_tok);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                       prompt_toks.data(), n_tok, add_bos, true) < 0) {
        die("tokenize");
    }
    printf("prompt tokens: %d\n", n_tok);
    if (n_tok + n_predict > n_ctx) {
        die("n_ctx too small for prompt + n_predict");
    }

    // =====================================================================
    // 2) 选设备, 建 backend. 干什么: 申请 CUDA 资源 (context + stream).
    //    本例强制走 CUDA: 找不到独立 GPU 就直接退出, 不退回 CPU.
    //    dev 是设备句柄, backend 是执行器.
    //    资源: 建 CUDA context 会占一小块固定显存. CPU 开销: 初始化, 一次性.
    // =====================================================================
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        die("no CUDA GPU found (this example requires CUDA)");
    }
    {
        // 确认这个 GPU 后端确实是 CUDA (不是 Vulkan/其它)
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";
        if (!reg_name || strcmp(reg_name, "CUDA") != 0) {
            die("GPU backend is not CUDA (this example requires CUDA)");
        }
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    printf("device: %s (%s)\n", props.name, props.description ? props.description : "");

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        die("backend init");
    }
    print_vram(dev, "after backend init");

    // =====================================================================
    // 3) 解析 GGUF + 权重上 GPU. 这是整个加载最重的一步.
    //    先读超参 (留 CPU) -> 建所有权重/KV 的张量头 (no_alloc, 只 metadata)
    //    -> ggml_backend_alloc_ctx_tensors 一次性在 GPU 开显存
    //    -> upload_tensors 把量化字节 cudaMemcpy 进去.
    //    no_alloc=true: gguf 只读目录, 不把 4GB blob 映射进内存.
    //    资源: 这里申请几个 GB 显存 (权重) + KV cache. CPU 开销: 磁盘 IO 是大头.
    // =====================================================================
    struct gguf_init_params gp = { true, nullptr };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) {
        die("gguf open");
    }
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id < 0) {
        die("no general.architecture");
    }
    const char * arch = gguf_get_val_str(gguf, arch_id);
    if (strcmp(arch, "qwen2") != 0) {
        fprintf(stderr, "this example only builds a qwen2 graph (got %s)\n", arch);
        return 1;
    }

    qwen_model model = {};
    model.n_ctx = n_ctx;
    model.hparams.n_embd       = kv_u32(gguf, arch, "embedding_length");
    model.hparams.n_layer      = kv_u32(gguf, arch, "block_count");
    model.hparams.n_ff         = kv_u32(gguf, arch, "feed_forward_length");
    model.hparams.n_head       = kv_u32(gguf, arch, "attention.head_count");
    {
        const int64_t id = kv_find(gguf, arch, "attention.head_count_kv");
        model.hparams.n_head_kv = id >= 0 ? gguf_get_val_u32(gguf, id) : model.hparams.n_head;
    }
    model.hparams.n_ctx_train  = kv_u32(gguf, arch, "context_length");
    model.hparams.rms_eps      = kv_f32_or(gguf, arch, "attention.layer_norm_rms_epsilon", 1e-6f);
    model.hparams.rope_freq_base = kv_f32_or(gguf, arch, "rope.freq_base", 10000.0f);
    model.hparams.n_rot        = model.hparams.n_embd / model.hparams.n_head;
    model.hparams.n_vocab      = (uint32_t) llama_vocab_n_tokens(vocab);

    printf("qwen2: n_layer=%u n_embd=%u n_head=%u n_head_kv=%u n_ff=%u n_rot=%u rope_base=%.0f n_vocab=%u\n",
           model.hparams.n_layer, model.hparams.n_embd, model.hparams.n_head,
           model.hparams.n_head_kv, model.hparams.n_ff, model.hparams.n_rot,
           model.hparams.rope_freq_base, model.hparams.n_vocab);

    // ctx_w 只给张量头留空间, 真正的字节在 GPU buffer 里
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    const size_t ctx_w_size = ggml_tensor_overhead() * (size_t) (n_tensors + 2 * model.hparams.n_layer + 32);
    struct ggml_init_params ip = { ctx_w_size, nullptr, true };
    model.ctx_w = ggml_init(ip);

    model.tok_embd    = add_weight(gguf, model.ctx_w, "token_embd.weight");
    model.output_norm = add_weight(gguf, model.ctx_w, "output_norm.weight");
    if (gguf_find_tensor(gguf, "output.weight") >= 0) {
        model.output = add_weight(gguf, model.ctx_w, "output.weight");
    } else {
        model.output = model.tok_embd; // 部分模型 lm_head 和 embedding 共享
    }
    model.hparams.n_vocab = (uint32_t) model.output->ne[1];

    const int n_embd_gqa = (int) (model.hparams.n_rot * model.hparams.n_head_kv);
    model.layers.resize(model.hparams.n_layer);
    for (uint32_t i = 0; i < model.hparams.n_layer; ++i) {
        char name[128];
        qwen_layer & L = model.layers[i];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);     L.attn_norm = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);        L.wq = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_q.bias", i);          L.bq = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);        L.wk = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_k.bias", i);          L.bk = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);        L.wv = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_v.bias", i);          L.bv = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);   L.wo = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);      L.ffn_norm = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);      L.ffn_gate = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);        L.ffn_up = add_weight(gguf, model.ctx_w, name);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);      L.ffn_down = add_weight(gguf, model.ctx_w, name);
        // KV cache: 每层存历史 token 的 K/V, 避免每步重算前文. 形状 [n_embd_gqa, n_ctx].
        L.k_cache = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, n_embd_gqa, n_ctx);
        L.v_cache = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, n_embd_gqa, n_ctx);
        snprintf(name, sizeof(name), "k_cache.%u", i); ggml_set_name(L.k_cache, name);
        snprintf(name, sizeof(name), "v_cache.%u", i); ggml_set_name(L.v_cache, name);
    }

    printf("alloc weights + KV on device...\n");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(model.ctx_w, backend);
    if (!wbuf) {
        die("GPU alloc failed");
    }
    printf("  weights+KV buffer: %s\n", human_size((double) ggml_backend_buffer_get_size(wbuf)).c_str());
    print_vram(dev, "after weights+KV");
    upload_tensors(gguf, model.ctx_w, path.c_str());
    {
        // KV cache 清零, 未写入的槽被 mask 挡住, 但不要留随机显存垃圾
        const std::vector<float> z(n_embd_gqa * n_ctx, 0.0f);
        for (uint32_t i = 0; i < model.hparams.n_layer; ++i) {
            ggml_backend_tensor_set(model.layers[i].k_cache, z.data(), 0, z.size() * sizeof(float));
            ggml_backend_tensor_set(model.layers[i].v_cache, z.data(), 0, z.size() * sizeof(float));
        }
    }
    gguf_free(gguf);
    printf("KV cache: %s\n", human_size((double) n_embd_gqa * n_ctx * 2 * 4 * model.hparams.n_layer).c_str());

    // gallocr: 计算图中间激活的显存分配器 (RMSNorm/QKV/softmax 的临时 buffer).
    // 和权重那块分开: 权重常驻, 中间激活可复用. 这里只是建分配器, 还没真分配.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // =====================================================================
    // 4) 建图 + 给中间激活分配显存. 干什么: 准备好可反复执行的前向图.
    //    图只建一次, 每步只改 token/pos/idx/mask 这几个 input, 再 graph_compute.
    //    若每步都新建图, gallocr 会以为 n_nodes 变了, Debug 下狂刷 realloc 日志.
    //    资源: ggml_gallocr_alloc_graph 在此申请一块中间激活显存 (远小于权重).
    //    CPU 开销: 建图/分配一次性, 可忽略.
    // =====================================================================
    const int n_graph_nodes = 8192;
    const size_t buf_size = ggml_tensor_overhead() * n_graph_nodes + ggml_graph_overhead_custom(n_graph_nodes, false);
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params cp = { buf.size(), buf.data(), true };
    ggml_context * ctx_c = ggml_init(cp);

    ggml_tensor * t_tok  = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, 1);
    ggml_tensor * t_pos  = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, 1);
    ggml_tensor * t_idx  = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I64, 1);
    ggml_tensor * t_mask = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, n_ctx, 1);
    ggml_set_input(t_tok);
    ggml_set_input(t_pos);
    ggml_set_input(t_idx);
    ggml_set_input(t_mask);
    ggml_set_name(t_tok, "tokens");
    ggml_set_name(t_pos, "pos");
    ggml_set_name(t_idx, "kv_idx");
    ggml_set_name(t_mask, "mask");

    ggml_cgraph * gf = build_qwen_graph(ctx_c, model, t_tok, t_pos, t_mask, t_idx);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        die("alloc graph");
    }
    printf("  compute buffer: %s\n", human_size((double) ggml_gallocr_get_buffer_size(galloc, 0)).c_str());
    print_vram(dev, "after graph alloc");
    ggml_tensor * logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t) {
        die("logits node missing");
    }
    printf("graph nodes: %d  logits: [%lld, %lld]\n",
           ggml_graph_n_nodes(gf), (long long) logits_t->ne[0], (long long) logits_t->ne[1]);

    // decode_one: 跑一个 token 的完整前向, 返回贪心选出的下一个 token.
    // 干什么:
    //   1) 写 4 个 input (当前 token / 位置 / KV 写入行号 / 因果 mask) 到 GPU
    //   2) graph_compute: 真正发射 CUDA kernel, 跑完 28 层拿到 logits
    //   3) 取回 logits 到 CPU, argmax 选分数最高的 token
    // 变量: tok=当前输入 token, pos=它在序列里的绝对位置 (也是 KV 写入行).
    //   mask: <=pos 可见 (0), >pos 屏蔽 (-inf), 实现因果注意力.
    // CPU 开销: 只有写 input + argmax(152064 次比较) 在 CPU, 主算力全在 GPU.
    //   GPU 每步要过 28 层矩阵乘, 是推理的真正瓶颈.
    std::vector<float> kq_mask(n_ctx, 0.0f);
    std::vector<float> logits((size_t) logits_t->ne[0]);
    auto decode_one = [&](llama_token tok, int pos) -> llama_token {
        const int32_t tok_i = tok;
        const int32_t pos_i = pos;
        const int64_t idx_i = pos;
        for (int i = 0; i < n_ctx; ++i) {
            kq_mask[i] = (i <= pos) ? 0.0f : -INFINITY;
        }
        ggml_backend_tensor_set(t_tok, &tok_i, 0, sizeof(tok_i));
        ggml_backend_tensor_set(t_pos, &pos_i, 0, sizeof(pos_i));
        ggml_backend_tensor_set(t_idx, &idx_i, 0, sizeof(idx_i));
        ggml_backend_tensor_set(t_mask, kq_mask.data(), 0, kq_mask.size() * sizeof(float));

        // 真正跑 GPU: 这一句是每步最耗时的地方 (28 层矩阵乘)
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            die("graph compute");
        }

        ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));

        // 贪心采样: 直接取最大 logit. 没有 top-k/温度, 所以输出是确定的.
        int best = 0;
        for (int i = 1; i < (int) logits.size(); ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return (llama_token) best;
    };

    // =====================================================================
    // 5) 推理主循环. 分两段:
    //    prefill: 把 prompt 逐 token 喂进去, 写满 KV cache (最后一个的输出是首个预测).
    //    decode:  自回归, 每次拿上一个 token 生成下一个, 直到 eos 或到 n_predict.
    //    CPU 开销: 循环本身很轻; 真正的时间花在每次 decode_one 里的 GPU 计算.
    // =====================================================================
    printf("<<<PROMPT_START>>>\n%s\n<<<PROMPT_END>>>\n", prompt.c_str());
    fflush(stdout);

    // prefill: 逐 token 过一遍, 只为写 KV; last 是最后一步预测出的首个新 token
    int pos = 0;
    llama_token last = 0;
    for (; pos < n_tok; ++pos) {
        last = decode_one(prompt_toks[pos], pos);
    }

    printf("<<<OUTPUT_START>>>\n");
    fflush(stdout);

    std::string response;
    int n_gen = 0;
    bool hit_eog = false;
    for (; n_gen < n_predict && pos < n_ctx; ++n_gen, ++pos) {
        if (llama_vocab_is_eog(vocab, last)) {
            hit_eog = true;
            break;
        }
        char piece[256];
        const int n = llama_token_to_piece(vocab, last, piece, sizeof(piece), 0, true);
        if (n > 0) {
            fwrite(piece, 1, n, stdout);
            fflush(stdout);
            response.append(piece, n);
        }
        last = decode_one(last, pos);
    }

    const char * stop = hit_eog ? "eos" : "n_predict";
    printf("\n<<<OUTPUT_END>>> tokens=%d stop=%s\n", n_gen, stop);
    printf("<<<RESULT_START>>>\n%s\n<<<RESULT_END>>>\n", response.c_str());
    fflush(stdout);

    // 收尾: 释放前面申请的所有资源 (图 buffer / 分配器 / 权重显存 / backend / 词表).
    // 顺序上先释放依赖方, 再释放被依赖方. 进程退出其实也会回收, 这里显式做一遍.
    ggml_free(ctx_c);
    ggml_gallocr_free(galloc);
    ggml_free(model.ctx_w);
    ggml_backend_free(backend);
    llama_model_free(vocab_model);
    llama_backend_free();
    return 0;
}
