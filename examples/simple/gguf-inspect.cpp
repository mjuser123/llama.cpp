/**
 * gguf-inspect.cpp
 *
 * Parse a GGUF file and print each part:
 *   what it is, what it is used for, where it lives at inference time.
 *
 * Does not load tensor blobs into RAM. Metadata only.
 *
 * 根目录下执行:
 *   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
 *   cmake --build build-cuda --target gguf-inspect
 *   ./build-cuda/bin/gguf-inspect -m model.gguf
 */

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

struct explain {
    const char * key;   // exact, or suffix after "arch."
    const char * role;
    const char * dest;
};

// KV: role + dest after llama_model_load
static const explain KV_TABLE[] = {
    { "general.architecture",             "模型结构名, 决定用哪套计算图",                     "CPU: llama_model.arch" },
    { "general.name",                     "显示名",                                           "CPU: 仅打印" },
    { "general.type",                     "文件种类 (model / adapter / ...)",                 "CPU: 加载时校验" },
    { "general.quantization_version",     "量化格式版本",                                     "CPU: 加载时校验" },
    { "general.file_type",                "整体量化类型 (Q4_K_M, F16, ...)",                  "CPU: llama_model.ftype" },
    { "general.alignment",                "权重 blob 对齐, 默认 32",                          "CPU: 映射权重时用" },
    { "general.license",                  "许可证",                                           "CPU: 仅打印" },
    { "general.url",                      "模型链接",                                         "CPU: 仅打印" },
    { "general.description",              "描述",                                             "CPU: 仅打印" },
    { "general.author",                   "作者",                                             "CPU: 仅打印" },
    { "general.version",                  "版本字符串",                                       "CPU: 仅打印" },
    { "general.source.url",               "上游来源",                                         "CPU: 仅打印" },
    { "general.source.huggingface.repository", "HF 仓库 id",                                  "CPU: 仅打印" },
    { "general.organization",             "组织名",                                           "CPU: 仅打印" },
    { "general.basename",                 "模型基名",                                         "CPU: 仅打印" },
    { "general.size_label",               "规模标签 (7B / 70B)",                              "CPU: 仅打印" },

    { "context_length",                   "训练上下文长度 n_ctx_train",                       "CPU: llama_hparams. 默认 n_ctx; 超过会警告, 不硬卡 KV cache" },
    { "embedding_length",                 "隐藏维度 n_embd",                                  "CPU: llama_hparams; 决定权重形状和 KV 宽度" },
    { "block_count",                      "Transformer 层数 n_layer",                         "CPU: llama_hparams; 搭图循环 + KV 层数" },
    { "feed_forward_length",              "FFN 中间维度 n_ff",                                "CPU: llama_hparams.n_ff" },
    { "vocab_size",                       "词表大小",                                         "CPU: llama_vocab + token_embd 行数" },
    { "attention.head_count",             "Q 头数 n_head",                                    "CPU: llama_hparams.n_head" },
    { "attention.head_count_kv",          "KV 头数 n_head_kv (小于 n_head 就是 GQA)",         "CPU: llama_hparams; 决定 KV cache 宽度" },
    { "attention.key_length",             "每个头的 K 维度",                                  "CPU: llama_hparams.n_embd_head_k" },
    { "attention.value_length",           "每个头的 V 维度",                                  "CPU: llama_hparams.n_embd_head_v" },
    { "attention.layer_norm_rms_epsilon", "RMSNorm eps",                                      "CPU: 传给 ggml_rms_norm" },
    { "attention.layer_norm_epsilon",     "LayerNorm eps",                                    "CPU: 传给 ggml_norm" },
    { "attention.causal",                 "是否因果 mask",                                    "CPU: 构造 attention mask" },
    { "attention.sliding_window",         "滑动窗口大小",                                     "CPU: SWA mask / KV" },
    { "rope.freq_base",                   "RoPE 基频 theta",                                  "CPU: 运行时生成 RoPE 频率表" },
    { "rope.dimension_count",             "RoPE 作用维度",                                    "CPU: llama_hparams.n_rot" },
    { "rope.scaling.type",                "RoPE 缩放方式 (linear/yarn/...)",                  "CPU: llama_hparams rope 字段" },
    { "rope.scaling.factor",              "RoPE 缩放系数",                                    "CPU: llama_hparams rope 字段" },
    { "rope.scaling.original_context_length", "缩放前的上下文长度",                           "CPU: llama_hparams rope 字段" },
    { "expert_count",                     "MoE expert 总数",                                  "CPU: llama_hparams.n_expert" },
    { "expert_used_count",                "每个 token 用几个 expert",                         "CPU: llama_hparams.n_expert_used" },

    { "tokenizer.ggml.model",             "分词算法 (llama / gpt2 / spm)",                    "CPU: llama_vocab, 不上 GPU" },
    { "tokenizer.ggml.pre",               "pre-tokenizer (qwen2 / llama-bpe / ...)",          "CPU: llama_vocab" },
    { "tokenizer.ggml.tokens",            "id -> 字符串 词表",                                "CPU: llama_vocab; detokenize" },
    { "tokenizer.ggml.scores",            "token 分数 (spm / 部分 bpe)",                      "CPU: llama_vocab" },
    { "tokenizer.ggml.merges",            "BPE merge 规则",                                   "CPU: llama_vocab; tokenize" },
    { "tokenizer.ggml.token_type",        "normal / control / user-defined / unused",         "CPU: llama_vocab" },
    { "tokenizer.ggml.bos_token_id",      "BOS id",                                           "CPU: llama_vocab" },
    { "tokenizer.ggml.eos_token_id",      "EOS id",                                           "CPU: llama_vocab; 生成结束" },
    { "tokenizer.ggml.eot_token_id",      "EOT id",                                           "CPU: llama_vocab" },
    { "tokenizer.ggml.unknown_token_id",  "UNK id",                                           "CPU: llama_vocab" },
    { "tokenizer.ggml.padding_token_id",  "PAD id",                                           "CPU: llama_vocab" },
    { "tokenizer.ggml.add_bos_token",     "是否自动加 BOS",                                   "CPU: tokenize()" },
    { "tokenizer.ggml.add_eos_token",     "是否自动加 EOS",                                   "CPU: tokenize()" },
    { "tokenizer.ggml.add_space_prefix",  "句首空格规则",                                     "CPU: tokenize()" },
    { "tokenizer.chat_template",          "jinja 聊天模板",                                   "CPU: 分词前把对话格式化成文本" },

    { "split.no",                         "当前分片序号",                                     "CPU: 多文件加载" },
    { "split.count",                      "分片总数",                                         "CPU: 多文件加载" },
    { "split.tensors.count",              "所有分片的张量总数",                               "CPU: 多文件加载" },
};

static const explain TENSOR_TABLE[] = {
    { "token_embd",     "词嵌入: token id 查成 n_embd 向量",
      "权重. 先 mmap/RAM, offload 后进 GPU. 计算图第一个节点 (ggml_get_rows)" },
    { "token_embd_norm","词嵌入上的 norm",
      "权重. 和 token_embd 同一块设备" },
    { "pos_embd",       "可学习位置编码 (少见, 多数模型用 RoPE)",
      "权重. 跟 embedding 一起放 CPU 或 GPU" },
    { "output_norm",    "lm_head 前的最终 RMSNorm",
      "权重. 一般在 GPU (输出张量)" },
    { "output",         "lm_head: n_embd -> 词表 logits",
      "权重. GPU. 有的模型与 token_embd 绑定, 此时没有这个张量" },
    { "rope_freqs",     "缓存的 RoPE 频率 (可选)",
      "小表, 常留 CPU; 很多模型运行时现算" },
    { "attn_norm",      "Attention 前的 RMSNorm gamma",
      "每层权重. 该层被 n_gpu_layers 覆盖则进 GPU" },
    { "attn_q.bias",    "Attention Q bias",
      "每层权重. GPU. 加在 Wq 输出上 (Qwen2 有, Llama 通常没有)" },
    { "attn_k.bias",    "Attention K bias",
      "每层权重. GPU. 加在 Wk 输出上, 再写入 KV cache" },
    { "attn_v.bias",    "Attention V bias",
      "每层权重. GPU. 加在 Wv 输出上, 再写入 KV cache" },
    { "attn_q",         "Attention Wq",
      "每层权重. GPU. 和隐状态做 ggml_mul_mat" },
    { "attn_k",         "Attention Wk",
      "每层权重. GPU. 结果写入 KV cache 的 K" },
    { "attn_v",         "Attention Wv",
      "每层权重. GPU. 结果写入 KV cache 的 V" },
    { "attn_output",    "Attention Wo",
      "每层权重. GPU. 把 attention 输出投回 n_embd" },
    { "attn_qkv",       "融合的 QKV",
      "每层权重. GPU. 乘完再拆成 Q/K/V" },
    { "attn_q_norm",    "Q 的 RMSNorm (Qwen3 等)",
      "每层权重. 跟该层一起进 GPU" },
    { "attn_k_norm",    "K 的 RMSNorm",
      "每层权重. 跟该层一起进 GPU" },
    { "ffn_norm",       "FFN 前的 RMSNorm gamma",
      "每层权重. 该层 offload 则进 GPU" },
    { "ffn_gate",       "FFN gate (SiLU/SwiGLU 左支)",
      "每层权重. GPU. 通常是除 attention 外最大的 matmul" },
    { "ffn_up",         "FFN up",
      "每层权重. GPU" },
    { "ffn_down",       "FFN down, 投回 n_embd",
      "每层权重. GPU" },
    { "ffn_gate_inp",   "MoE 路由器",
      "每层权重. GPU" },
    { "ffn_gate_exps",  "MoE expert gate",
      "每层权重. GPU, 体积大" },
    { "ffn_up_exps",    "MoE expert up",
      "每层权重. GPU, 体积大" },
    { "ffn_down_exps",  "MoE expert down",
      "每层权重. GPU, 体积大" },
};

static std::string human_size(double bytes) {
    const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    int u = 0;
    double v = bytes;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return std::string(buf);
}

static uint64_t path_size(const char * path) {
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path, &st) != 0) {
        return 0;
    }
    return (uint64_t) st.st_size;
#else
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return (uint64_t) st.st_size;
#endif
}

static std::string escape_str(const char * s, size_t maxn) {
    if (!s) {
        return "(null)";
    }
    std::string out;
    size_t i = 0;
    for (; s[i] && out.size() < maxn; ++i) {
        const unsigned char c = (unsigned char) s[i];
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 32) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\x%02x", c);
            out += tmp;
        } else {
            out += (char) c;
        }
    }
    if (s[i]) {
        out += "...";
    }
    return out;
}

static bool key_matches(const char * key, const char * pat, const char * arch) {
    if (strcmp(key, pat) == 0) {
        return true;
    }
    const size_t pk = strlen(key);
    const size_t pp = strlen(pat);
    if (pk > pp && key[pk - pp - 1] == '.' && strcmp(key + pk - pp, pat) == 0) {
        return true;
    }
    if (arch && arch[0]) {
        std::string prefixed = std::string(arch) + "." + pat;
        if (strcmp(key, prefixed.c_str()) == 0) {
            return true;
        }
    }
    return false;
}

static const explain * find_kv(const char * key, const char * arch) {
    const explain * best = nullptr;
    size_t best_len = 0;
    for (const explain & e : KV_TABLE) {
        if (key_matches(key, e.key, arch)) {
            const size_t n = strlen(e.key);
            if (n >= best_len) {
                best = &e;
                best_len = n;
            }
        }
    }
    return best;
}

static const explain * find_tensor(const char * name) {
    const explain * best = nullptr;
    size_t best_len = 0;
    for (const explain & e : TENSOR_TABLE) {
        if (strstr(name, e.key) && strlen(e.key) >= best_len) {
            best = &e;
            best_len = strlen(e.key);
        }
    }
    return best;
}

static std::string kv_value_str(const gguf_context * ctx, int64_t i, size_t max_chars) {
    const enum gguf_type t = gguf_get_kv_type(ctx, i);
    char buf[256];

    switch (t) {
        case GGUF_TYPE_UINT8:  snprintf(buf, sizeof(buf), "%u",   (unsigned) gguf_get_val_u8 (ctx, i)); return buf;
        case GGUF_TYPE_INT8:   snprintf(buf, sizeof(buf), "%d",   (int)      gguf_get_val_i8 (ctx, i)); return buf;
        case GGUF_TYPE_UINT16: snprintf(buf, sizeof(buf), "%u",   (unsigned) gguf_get_val_u16(ctx, i)); return buf;
        case GGUF_TYPE_INT16:  snprintf(buf, sizeof(buf), "%d",   (int)      gguf_get_val_i16(ctx, i)); return buf;
        case GGUF_TYPE_UINT32: snprintf(buf, sizeof(buf), "%u",   (unsigned) gguf_get_val_u32(ctx, i)); return buf;
        case GGUF_TYPE_INT32:  snprintf(buf, sizeof(buf), "%d",              gguf_get_val_i32(ctx, i)); return buf;
        case GGUF_TYPE_FLOAT32:snprintf(buf, sizeof(buf), "%.6g",            gguf_get_val_f32(ctx, i)); return buf;
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx, i) ? "true" : "false";
        case GGUF_TYPE_STRING: return escape_str(gguf_get_val_str(ctx, i), max_chars);
        case GGUF_TYPE_UINT64: snprintf(buf, sizeof(buf), "%llu", (unsigned long long) gguf_get_val_u64(ctx, i)); return buf;
        case GGUF_TYPE_INT64:  snprintf(buf, sizeof(buf), "%lld", (long long)          gguf_get_val_i64(ctx, i)); return buf;
        case GGUF_TYPE_FLOAT64:snprintf(buf, sizeof(buf), "%.6g",                      gguf_get_val_f64(ctx, i)); return buf;
        case GGUF_TYPE_ARRAY: {
            const enum gguf_type at = gguf_get_arr_type(ctx, i);
            const size_t n = gguf_get_arr_n(ctx, i);
            std::string s = std::string("[") + gguf_type_name(at) + " x " + std::to_string(n) + "]";
            if (at == GGUF_TYPE_STRING && n > 0) {
                s += " e.g. \"";
                s += escape_str(gguf_get_arr_str(ctx, i, 0), 24);
                s += "\"";
                if (n > 1) {
                    s += ", \"";
                    s += escape_str(gguf_get_arr_str(ctx, i, 1), 24);
                    s += "\"";
                }
            }
            return s;
        }
        default:
            return "?";
    }
}

static const char * kv_group(const char * key, const char * arch) {
    if (strncmp(key, "general.", 8) == 0) {
        return "身份 / 文件信息";
    }
    if (strncmp(key, "tokenizer.", 10) == 0) {
        return "分词器 (只在 CPU)";
    }
    if (strncmp(key, "split.", 6) == 0) {
        return "多文件分片";
    }
    if (arch && arch[0] && strncmp(key, arch, strlen(arch)) == 0 && key[strlen(arch)] == '.') {
        return "结构超参 (用来搭计算图)";
    }
    return "其它";
}

static std::string shape_str(const int64_t * ne) {
    char buf[128];
    snprintf(buf, sizeof(buf), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
             ne[0], ne[1], ne[2], ne[3]);
    return buf;
}

static std::string canon_tensor(const char * name) {
    // blk.12.attn_q.weight -> blk.*.attn_q.weight
    const char * p = name;
    if (strncmp(p, "blk.", 4) == 0) {
        p += 4;
        while (*p && *p != '.') {
            ++p;
        }
        if (*p == '.') {
            return std::string("blk.*") + p;
        }
    }
    return std::string(name);
}

static void print_usage(const char * argv0) {
    printf("usage: %s -m model.gguf\n", argv0);
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string path;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            path = argv[++i];
        } else if (path.empty() && argv[i][0] != '-') {
            path = argv[i];
        }
    }
    if (path.empty()) {
        path = "models/qwen2.5-7b-instruct-q3_k_m.gguf";
        printf("未传 -m, 尝试默认路径: %s\n", path.c_str());
    }

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "无法打开 GGUF: %s\n", path.c_str());
        print_usage(argv[0]);
        return 1;
    }

    const uint64_t fsz        = path_size(path.c_str());
    const uint32_t version    = gguf_get_version(ctx);
    const size_t   alignment  = gguf_get_alignment(ctx);
    const size_t   data_off   = gguf_get_data_offset(ctx);
    const int64_t  n_kv       = gguf_get_n_kv(ctx);
    const int64_t  n_tensors  = gguf_get_n_tensors(ctx);

    const char * arch = "";
    {
        const int64_t k = gguf_find_key(ctx, "general.architecture");
        if (k >= 0 && gguf_get_kv_type(ctx, k) == GGUF_TYPE_STRING) {
            arch = gguf_get_val_str(ctx, k);
        }
    }

    printf("\n========== GGUF 文件布局 ==========\n");
    printf("路径           : %s\n", path.c_str());
    printf("文件大小       : %s (%llu bytes)\n", human_size((double) fsz).c_str(),
           (unsigned long long) fsz);
    printf("\n");
    printf("  偏移 0       文件头     magic \"GGUF\" + version + n_tensors + n_kv\n");
    printf("  接着         KV 元数据  超参/词表/名字, 不含权重\n");
    printf("  接着         张量目录   名字 / 形状 / 类型 / 偏移  (仍不含权重)\n");
    printf("  然后填充     对齐到 %zu 字节\n", alignment);
    printf("  偏移 %-6zu 权重 blob  所有权重字节拼在一起\n", data_off);
    printf("\n");
    printf("version        : %u  (当前规范是 %d)\n", version, GGUF_VERSION);
    printf("n_kv           : %" PRId64 "\n", n_kv);
    printf("n_tensors      : %" PRId64 "\n", n_tensors);
    printf("alignment      : %zu\n", alignment);
    printf("data offset    : %zu  (文件头+KV+目录+填充 = %s)\n",
           data_off, human_size((double) data_off).c_str());
    if (fsz >= data_off) {
        printf("blob size      : %s\n", human_size((double) (fsz - data_off)).c_str());
    }
    printf("architecture   : %s\n", arch[0] ? arch : "(缺少 general.architecture)");
    printf("\n");
    printf("文件头落点     : 解析完就丢掉, 值拷进 C 结构体.\n");

    printf("\n========== 1) KV 元数据  (n=%" PRId64 ") ==========\n", n_kv);
    printf("这些不是权重. 加载器用它们填 llama_hparams / llama_vocab, 并决定怎么搭 ggml 图.\n");
    printf("全部留在 CPU, 不会上 GPU.\n\n");

    const char * last_group = nullptr;
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        const char * grp = kv_group(key, arch);
        if (grp != last_group) {
            printf("--- %s ---\n", grp);
            last_group = grp;
        }

        const enum gguf_type t = gguf_get_kv_type(ctx, i);
        const std::string val = kv_value_str(ctx, i, 80);
        const explain * ex = find_kv(key, arch);

        printf("  [%3" PRId64 "] %-44s  %-8s  %s\n",
               i, key, gguf_type_name(t), val.c_str());
        if (ex) {
            printf("        作用 : %s\n", ex->role);
            printf("        落点 : %s\n", ex->dest);
        } else {
            printf("        作用 : (未列入说明表, 仍是 CPU 元数据)\n");
            printf("        落点 : CPU, 忽略或存成 extra kv\n");
        }
    }

    printf("\n========== 2) 张量目录  (n=%" PRId64 ") ==========\n", n_tensors);
    printf("下面每条是文件里的 ggml_tensor 头.\n");
    printf("权重字节在: data_offset + tensor_offset, 先 mmap/读入,\n");
    printf("再由 ggml_backend_tensor_set 按 n_gpu_layers 拷到 CPU RAM 或 GPU VRAM.\n\n");

    struct kind_stat {
        uint64_t n = 0;
        uint64_t bytes = 0;
        const explain * ex = nullptr;
    };
    std::map<std::string, kind_stat> kinds;
    uint64_t total_tensor_bytes = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        const size_t nb   = gguf_get_tensor_size(ctx, i);
        total_tensor_bytes += nb;
        const std::string ck = canon_tensor(name);
        kind_stat & st = kinds[ck];
        st.n += 1;
        st.bytes += nb;
        if (!st.ex) {
            st.ex = find_tensor(name);
        }
    }

    printf("--- 张量种类 (blk.N 收成 blk.*) ---\n");
    for (const auto & kv : kinds) {
        printf("  %-36s  %4llu x  %10s\n",
               kv.first.c_str(),
               (unsigned long long) kv.second.n,
               human_size((double) kv.second.bytes).c_str());
        if (kv.second.ex) {
            printf("      作用 : %s\n", kv.second.ex->role);
            printf("      落点 : %s\n", kv.second.ex->dest);
        } else {
            printf("      作用 : (未列入说明表, 仍是权重)\n");
            printf("      落点 : 权重 buffer, 按 n_gpu_layers 进 CPU 或 GPU\n");
        }
    }

    printf("\n--- 完整清单 ---\n");
    printf("  idx  type     bytes        file-offset   shape                    name\n");
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name   = gguf_get_tensor_name(ctx, i);
        const size_t nb     = gguf_get_tensor_size(ctx, i);
        const size_t off    = gguf_get_tensor_offset(ctx, i);
        const enum ggml_type ty = gguf_get_tensor_type(ctx, i);
        const int64_t * ne  = gguf_get_tensor_ne(ctx, i);

        printf("  [%3" PRId64 "] %-7s %10s  +%-12zu %-24s %s\n",
               i,
               ggml_type_name(ty),
               human_size((double) nb).c_str(),
               off,
               shape_str(ne).c_str(),
               name);
    }
    printf("\n权重合计       : %s  (各张量 size 之和)\n",
           human_size((double) total_tensor_bytes).c_str());

    printf("\n========== 3) 推理时落点 ==========\n");
    printf("\n");
    printf("  磁盘上的 GGUF\n");
    printf("    |-- 文件头 + KV + 张量目录\n");
    printf("    |     -> CPU 结构体: llama_hparams, llama_vocab, ggml_tensor 元数据\n");
    printf("    |        分词器永远不上 GPU\n");
    printf("    |\n");
    printf("    +-- 权重 blob\n");
    printf("          -> mmap 或读进主机内存\n");
    printf("          -> ggml_backend_tensor_set\n");
    printf("                |-- n_gpu_layers 覆盖这一层 -> GPU 显存 (CUDA buffer)\n");
    printf("                +-- 否则                    -> CPU 内存 (CPU buffer)\n");
    printf("\n");
    printf("  GGUF 里没有, llama_init_from_model 时另分配:\n");
    printf("    KV cache     GPU 显存 (或 CPU). 随 n_ctx * n_layer * n_embd_kv 增长\n");
    printf("    compute buf  一张 ggml 图的显存scratch\n");
    printf("    activations  图里的临时张量, 每个 token 复用\n");
    printf("    logits       通常在主机, vocab_size 个 float, 给采样器\n");
    printf("    RoPE freqs   由 rope.freq_base 现算, 很小, 常在 GPU\n");
    printf("\n");
    printf("  一次 decode 大致:\n");
    printf("    CPU tokenize(prompt) 用 llama_vocab\n");
    printf("    GPU: embed -> N x (RMSNorm, QKV, RoPE, attn+KV, O, RMSNorm, FFN) -> RMSNorm -> lm_head\n");
    printf("    CPU 从 logits 采样下一个 token\n");
    printf("\n");

    gguf_free(ctx);
    return 0;
}
