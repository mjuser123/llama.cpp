/**
 * simple-qwen2-cuda-raw.cu
 *
 * 本文件做什么
 * ------------
 * 和 simple-qwen2-cuda.cpp 目标一样 (Qwen2 单 token 自回归推理), 但更底层:
 * 完全不用 ggml 的算子和计算图, 所有运算自己写 CUDA kernel + cuBLAS 发起.
 *
 * 和 simple-qwen2-cuda.cpp 的分工对比
 *   simple-qwen2-cuda.cpp: 搭 ggml 图 -> ggml_backend_graph_compute -> ggml 内部发 kernel
 *   本文件:                自己 cudaMalloc -> 自己写 kernel -> 自己 <<<>>> 发射
 *
 * 还在复用的现成轮子 (这些和"推理数学"无关, 重复造没意义):
 *   llama 分词 / 聊天模板, gguf 读元数据和权重目录, ggml 的类型转换 (反量化)
 *
 * 精度选择
 *   权重 + KV cache: F16. 这是显存大头, 减半.
 *   激活 (残差流/softmax/norm): F32. RMSNorm 求平方和、softmax 求指数在 F16 下
 *   容易溢出或掉精度, 真实引擎也是这么混着用的.
 *
 * 矩阵乘为什么用 cuBLAS
 *   单 token 解码时 matmul 退化成 矩阵 x 向量 (GEMV), 是访存瓶颈.
 *   手写一个能打的 GEMM 是另一个大课题, 和"理解推理流程"无关, 所以交给 cuBLAS.
 *   其余 7 个算子 (embedding/RMSNorm/RoPE/attention/SwiGLU/残差/采样) 全部手写.
 *
 * 一次前向的调用链
 *   token id
 *     --k_get_row-----> 隐状态 x [n_embd]         (F32)
 *     --每层 x24/28--->
 *         k_rms_norm   -> xb [n_embd]             (F16, 喂给 cuBLAS)
 *         matvec       -> q/k/v                   (cuBLAS, F16 权重 F32 输出)
 *         k_rope_neox  -> 给 q/k 加位置信息
 *         k_store_kv   -> 写 KV cache 第 pos 行
 *         k_attn       -> softmax(QK^T/sqrt(d))V
 *         matvec(wo) + k_add                      (残差)
 *         k_rms_norm / matvec(gate,up) / k_silu_mul / matvec(down) + k_add
 *     --matvec(lm_head)-> logits [n_vocab]
 *     --argmax (CPU)--> 下一个 token
 *
 * 模型要求
 *   任意 qwen2 架构的 GGUF 都能读 (加载时统一转成 F16), 但 F16 展开后要放得进显存.
 *   7B 转 F16 约 15GB, 8GB 的卡放不下. 学习用 Qwen2.5-0.5B-Instruct-fp16 (约 1.2GB).
 *
 * 根目录下执行:
 *   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build-cuda --target simple-qwen2-cuda-raw
 *   ./build-cuda/bin/simple-qwen2-cuda-raw -m ~/qwen2.5-0.5b-instruct-fp16.gguf -p "who are you?" -c 512 -n 64
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
#include "gguf.h"
#include "llama.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// 基础设施: 错误检查 / 日志 / 打印
// -----------------------------------------------------------------------------

static void die(const char * msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

// CUDA 的 API 失败不会抛异常, 不检查就会拿到静默的错误结果, 所以每个调用都包一层.
#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        const cudaError_t err_ = (x);                                                    \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(err_),        \
                    __FILE__, __LINE__);                                                 \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(x)                                                                  \
    do {                                                                                 \
        const cublasStatus_t st_ = (x);                                                  \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cublas error %d at %s:%d\n", (int) st_, __FILE__, __LINE__); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// kernel 是异步发射的, 出错要等同步才看得到. 调试期在每步后面调一次.
static void check_last_kernel(const char * tag) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed (%s): %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

// ggml/llama 的 INFO 日志会混进模型输出, 只放 WARN/ERROR 到 stderr.
static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
        return;
    }
    fputs(text, stderr);
    fflush(stderr);
}

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

// 查整卡显存. 注意这是全卡 (含其它进程), 不是本进程用量.
static void print_vram(const char * tag) {
    size_t free_b  = 0;
    size_t total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("||| vram [%s]: used %s / %s  (free %s)\n", tag,
           human_size((double) (total_b - free_b)).c_str(),
           human_size((double) total_b).c_str(),
           human_size((double) free_b).c_str());
}

// -----------------------------------------------------------------------------
// CUDA kernel: 归约原语
//
// 归约 = 把一个 block 里所有线程各自的一个值合成一个值 (求和/求最大).
// RMSNorm 要平方和, softmax 要最大值和指数和, 都靠它.
// 两级: 先 warp 内用 __shfl 寄存器交换 (最快), 再把每个 warp 的结果放共享内存合并.
// -----------------------------------------------------------------------------

#define WARP_SIZE 32

// 蝶形归约 (butterfly reduction). 原理:
//   __shfl_xor_sync(mask, v, off) 让本线程读到 "lane 号异或 off" 那个线程的 v.
//   异或是对称的: lane a 读 a^off, 而 a^off 读 (a^off)^off = a. 两边互换, 同时加上.
//   off 取 16,8,4,2,1 共 5 轮, 每轮参与的独立分组数减半:
//     第 1 轮 lane0 和 lane16 互换, lane1 和 lane17 互换, ...
//     第 5 轮 lane0 和 lane1 互换, ...
//   5 轮后每个 lane 手里都是全 32 个值的总和 (不是只有 lane0, 省掉了广播).
//   log2(32) = 5 步完成 32 个数求和, 而且全程在寄存器里, 不碰共享内存, 最快.
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// 同上, 把加法换成取最大. 求和与取最大都满足结合律和交换律, 所以能这样折半归约.
__device__ __forceinline__ float warp_max(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

// 跨 warp 归约. warp 之间没有寄存器通路, 只能走共享内存, 所以分两级:
//   1) 每个 warp 内部先用 shuffle 归约出一个数 (32 个数变 1 个)
//   2) 每个 warp 的 lane0 把结果写进 smem[warp_id]
//   3) warp 0 再把这最多 32 个数 (一个 block 最多 1024 线程 = 32 warp) 归约一次
// 一个 256 线程的 block 就是 256 -> 8 -> 1, 总共两级.
// smem 至少要有 WARP_SIZE 个 float. 返回后 smem 可以被下一次归约复用.
template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float * smem) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int nw   = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < nw) ? smem[threadIdx.x] : (IS_MAX ? -INFINITY : 0.0f);
    __syncthreads(); // 全部读完, 才允许 warp 0 覆写 smem[0]
    if (wid == 0) {
        v = IS_MAX ? warp_max(v) : warp_sum(v);
        if (lane == 0) {
            smem[0] = v;
        }
    }
    __syncthreads();
    v = smem[0];
    __syncthreads();
    return v;
}

// -----------------------------------------------------------------------------
// CUDA kernel: 各个算子
// -----------------------------------------------------------------------------

// 1) embedding 查表: x = E[token]
//
// 原理: 数学上这一步是 x = E^T * onehot(token), 即用一个只有第 token 位是 1 的
// 向量去乘嵌入矩阵. 但 onehot 乘法的结果就等于"取出第 token 行", 其余全是乘 0,
// 所以实现上直接查表, 把 O(n_vocab * n_embd) 的乘法降成 O(n_embd) 的拷贝.
//
// 这个 n_embd 维向量就是该 token 的初始语义表示, 后面 24/28 层不断改写它.
// 表是训练出来的: 意思相近的词, 向量方向也相近.
__global__ void k_get_row(float * dst, const __half * table, int64_t row, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(table[row * n + i]);
    }
}

// 2) RMSNorm:  dst[i] = src[i] / sqrt(mean(src^2) + eps) * w[i]
//
// 原理: 残差流 x 每过一层都被加上新东西, 数值会越滚越大, 后面的矩阵乘就容易
// 饱和/溢出. 归一化把向量"长度"拉回固定尺度, 只保留方向信息.
//
//   RMS(x) = sqrt( (1/n) * sum_i x_i^2 )    向量的均方根, 相当于 L2 范数 / sqrt(n)
//   x_hat  = x / RMS(x)                     归一化后 RMS(x_hat) 恒等于 1
//   dst    = x_hat * w                      w 是可学习的每维增益
//
// 为什么不是 LayerNorm: LayerNorm 要先减均值再除标准差 (x - mean) / std.
// 实验发现减均值这一步对 Transformer 几乎没贡献, 去掉后少一次归约, 更快.
// 这就是 RMSNorm, Llama/Qwen 系全都用它.
//
// eps 的作用: 万一整个向量全是 0, 分母会变成 0. 加个 1e-6 兜底.
// w 的作用: 归一化把所有维度压成同一尺度, 但不同维度的重要性本来就不同,
//   w 让模型能重新学出"这一维该放大多少", 把表达能力还回来.
//
// 实现: rsqrtf 是硬件倒数平方根指令, 比 1.0f/sqrtf(x) 快, 精度够用.
// 只用 1 个 block: n_embd 才几千, 一个 block 内归约比跨 block 通信划算.
// 输出直接给 F16, 因为它的下一步一定是喂给 cuBLAS 做矩阵乘.
__global__ void k_rms_norm(__half * dst, const float * src, const float * w, int n, float eps) {
    __shared__ float smem[WARP_SIZE];

    // 每个线程跨步累加一部分平方和, 再 block 内归约成总和
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = src[i];
        acc += v * v;
    }
    const float sum   = block_reduce<false>(acc, smem);
    const float scale = rsqrtf(sum / n + eps); // = 1 / RMS(x)

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = __float2half(src[i] * scale * w[i]);
    }
}

// 3) RoPE (旋转位置编码), NeoX 变体, 原地改写 q 或 k.
//
// 要解决的问题: attention 算的是 q 和 k 的点积, 而点积和顺序无关.
// 不给模型位置信息的话, "狗咬人"和"人咬狗"在它眼里一模一样.
//
// 原理: 把向量每两维看成复平面上的一个点, 按位置 pos 旋转一个角度.
//   取一对分量 (x0, x1), 旋转 theta 角:
//     x0' = x0*cos(theta) - x1*sin(theta)
//     x1' = x0*sin(theta) + x1*cos(theta)
//   这就是标准的二维旋转矩阵 R(theta) 乘上去.
//
// 为什么旋转就能编码位置, 关键在这个恒等式:
//   <R(m*a) q, R(n*a) k> = <q, R((n-m)*a) k>
//   两个向量各自旋转后再点积, 结果只取决于 (n - m), 也就是它们的"相对距离".
//   所以模型学到的是相对位置关系, 而不是绝对下标. 这比早期的绝对位置嵌入好:
//   同样一句话出现在第 5 个词还是第 500 个词, 内部的相对关系完全一致.
//
// 频率的设计: theta_i = pos * freq_base^(-2i/n_rot), i 从 0 到 n_rot/2 - 1.
//   i = 0 时指数为 0, theta = pos, 转得最快 (高频), 相邻 token 就能区分开.
//   i 最大时 theta 约等于 pos/freq_base, 转得极慢 (低频), 要隔很远才转过一圈,
//   负责表达长距离关系.
//   一组从快到慢的频率叠在一起, 就像时钟的秒针分针时针, 组合起来能唯一确定位置.
//   freq_base 越大低频越慢, 能表示的上下文越长. Qwen2.5 用 1000000 (常见的是 10000).
//
// NeoX 的配对方式是 (i, i + n_rot/2), 不是相邻的 (2i, 2i+1). 配错了输出就是乱码.
//   这只是分量摆放顺序的约定差异, 数学上等价, 但必须和训练时一致.
//
// V 不加 RoPE: 位置信息只需要影响"谁该注意谁"(打分), 不该改变"注意到之后取什么内容".
__global__ void k_rope_neox(float * x, int pos, int n_rot, int n_heads, float freq_base) {
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (idx >= n_heads * half) {
        return;
    }
    const int h = idx / half; // 第几个头
    const int i = idx % half; // 该头内的第几对分量

    const float theta = pos * powf(freq_base, -2.0f * i / (float) n_rot);
    float s;
    float c;
    sincosf(theta, &s, &c); // 一条指令同时出 sin 和 cos, 比分开算快

    float * p = x + (size_t) h * n_rot;
    const float x0 = p[i];
    const float x1 = p[i + half];
    p[i]        = x0 * c - x1 * s;
    p[i + half] = x0 * s + x1 * c;
}

// 4) 写 KV cache: kc[pos] = k, vc[pos] = v
//
// 为什么能缓存, 靠的是因果注意力这个性质:
//   第 t 个 token 的 k_t / v_t 只由它自己的输入算出 (x_t 乘 Wk/Wv),
//   不依赖它后面的任何 token. 所以一旦算出来就永远不会变.
// 如果没有 cache, 生成第 100 个 token 时要把前 99 个的 k/v 全部重算一遍,
// 总代价是 O(n^2); 有了 cache 每步只算 1 个, 总代价降到 O(n).
//
// 代价是显存: 每层每个位置要存 2 * n_embd_gqa 个数, 长上下文时它会超过权重本身.
// 这也是 GQA (多个 Q 头共享一组 KV) 被发明出来的主要动机 -- 直接把 cache 缩小.
// 存成 F16 再省一半, 精度损失对 attention 打分基本无影响.
__global__ void k_store_kv(__half * kc, __half * vc, const float * k, const float * v, int pos, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t o = (size_t) pos * n + i;
        kc[o] = __float2half(k[i]);
        vc[o] = __float2half(v[i]);
    }
}

// 5) 多头注意力: out = softmax(q . K^T / sqrt(d)) . V
//
// 直觉: 当前 token 要从前文里"查资料". 三个角色:
//   q (query)  当前 token 想找什么
//   k (key)    每个历史 token 能提供什么, 用来和 q 匹配打分
//   v (value)  匹配上之后真正取走的内容
// 和数据库查询很像, 区别是这里不是精确匹配, 而是给每条历史一个权重后加权平均.
//
// 逐步拆解:
//
// (a) 打分 s_t = dot(q, k_t)
//     点积衡量两个向量的方向相似度: 方向越接近值越大. 所以 s_t 就是
//     "当前 token 对第 t 个历史 token 的关注程度". 模型通过训练 Wq/Wk
//     来决定什么样的内容之间该有高分.
//
// (b) 缩放 / sqrt(d)
//     若 q/k 各维近似独立、方差为 1, 那么 d 维点积的方差就是 d, 也就是
//     标准差 sqrt(d). d = 64 或 128 时分数会散到正负十几.
//     直接喂给 softmax 会让它极度尖锐 (几乎变成只选最大的那个 one-hot),
//     梯度也随之消失. 除以 sqrt(d) 把方差拉回 1, 分布才平缓可用.
//
// (c) softmax: a_t = exp(s_t) / sum_j exp(s_j)
//     把任意实数分数变成一组非负、且加起来等于 1 的权重, 即概率分布.
//     用 exp 而不是简单归一化, 是为了放大差距: 分数高一点, 权重高很多.
//
// (d) 输出 out = sum_t a_t * v_t
//     按权重把历史的 value 加权平均. 权重大的历史 token 贡献多.
//
// 因果性: 只遍历 t <= pos, 后面的 token 压根不参与. 所以不需要 mask 张量 --
//   ggml 那版要构造 -inf 的 mask, 是因为图的形状必须固定成整段 n_ctx.
//
// 多头: 把 n_embd 切成 n_head 份各自独立做上面这套. 不同头可以学不同的关注模式
//   (有的盯语法, 有的盯指代), 最后拼回来再过 Wo 融合. 头之间无依赖, 所以
//   这里一个 block 负责一个头, 天然并行.
//
// GQA (分组查询注意力): n_head 个 Q 头共享 n_head_kv 组 KV, 第 h 个 Q 头用
//   第 h/gqa 组. 0.5B 是 14 个 Q 头共享 2 组 KV, 即 7 个 Q 头一组 (连续分组,
//   不是交错). 目的就是把 KV cache 缩小到 1/7.
__global__ void k_attn(__half * out, const float * q, const __half * kc, const __half * vc,
                       float * att, int pos, int n_rot, int n_head_kv, int gqa, int n_ctx, float scale) {
    __shared__ float smem[WARP_SIZE];

    const int h         = blockIdx.x;
    const int hkv       = h / gqa;
    const int n_kv_embd = n_head_kv * n_rot;
    const int T         = pos + 1; // 可见的历史长度

    const float * qh = q + (size_t) h * n_rot;
    float *       a  = att + (size_t) h * n_ctx; // 本头的分数暂存区

    // 5.1 打分: a[t] = dot(q, k_t) / sqrt(d). 一个线程负责若干个历史位置.
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const __half * kt = kc + (size_t) t * n_kv_embd + hkv * n_rot;
        float s = 0.0f;
        for (int d = 0; d < n_rot; ++d) {
            s += qh[d] * __half2float(kt[d]);
        }
        a[t] = s * scale;
    }
    __syncthreads();

    // 5.2 softmax, 用"减最大值"的数值稳定写法.
    //   数学上 exp(s_t)/sum(exp(s_j)) == exp(s_t - m)/sum(exp(s_j - m)) 对任意 m 成立,
    //   因为分子分母同乘了 exp(-m), 约掉了. 取 m = max(s) 的好处是所有指数都 <= 0,
    //   exp 结果落在 (0, 1], 绝不会上溢; 而直接算 exp(s) 在 s 稍大 (>88) 时 F32 就 inf 了.
    //   代价是要多扫一遍求最大值, 所以这里是三趟: 求 max -> 求指数和 -> 归一化.
    float m = -INFINITY;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        m = fmaxf(m, a[t]);
    }
    m = block_reduce<true>(m, smem);

    float z = 0.0f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float e = __expf(a[t] - m);
        a[t] = e; // 原地存指数值, 省一块暂存区
        z += e;
    }
    z = block_reduce<false>(z, smem);
    const float inv = 1.0f / z; // 归一化留到最后一步乘, 省 T 次除法

    // 5.3 加权求和: out[d] = sum_t a[t] * v_t[d].
    //   注意这里按输出维度 d 切分线程, 而 5.1 是按历史位置 t 切分 --
    //   两段的并行维度不同, 所以中间必须有 block_reduce 里的 __syncthreads.
    //   输出 F16, 因为下一步是 wo 矩阵乘.
    for (int d = threadIdx.x; d < n_rot; d += blockDim.x) {
        float s = 0.0f;
        for (int t = 0; t < T; ++t) {
            s += a[t] * __half2float(vc[(size_t) t * n_kv_embd + hkv * n_rot + d]);
        }
        out[(size_t) h * n_rot + d] = __float2half(s * inv);
    }
}

// 6) SwiGLU 的门控: dst[i] = silu(gate[i]) * up[i], 逐元素.
//
// 整个 FFN 是:  down( silu(gate(x)) * up(x) )
//   gate 和 up 是两条并行的升维投影 (n_embd -> n_ff, 0.5B 是 896 -> 4864),
//   逐元素相乘后再由 down 降回 n_embd.
//
// 为什么要非线性: attention 只是对 value 做加权平均, 是线性操作.
//   若干个线性层叠起来还是线性的, 等价于一层, 表达能力不够.
//   FFN 提供非线性, 也是模型存"知识"的主要地方 (参数量约占三分之二).
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   相比 ReLU (负数直接砍成 0), silu 在负半轴留了一条平滑的小尾巴,
//   处处可导, 训练更稳, 效果也更好.
//
// 为什么要"门控"(乘 up 那一路): 普通 FFN 是 down(silu(up(x))), 只有一条路.
//   SwiGLU 多出一条 gate 分支当开关: silu(gate) 的值接近 0 就把 up 那一路
//   对应的维度关掉, 接近 1 就放行. 相当于让网络能按输入内容动态选择
//   激活哪些特征, 同参数量下效果明显更好. 代价是 FFN 权重从 2 个矩阵变 3 个.
__global__ void k_silu_mul(__half * dst, const float * gate, const float * up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        dst[i] = __float2half((g / (1.0f + __expf(-g))) * up[i]);
    }
}

// 7) 残差相加: x = x + y
//
// 每个子层算的不是"新的 x", 而是"要在 x 上改动多少", 即 x = x + F(x).
// 好处:
//   1) 前向上给了信息一条直通路, 浅层特征能原样传到深层, 不会被 24 层反复揉碎.
//   2) 反向上梯度可以沿着这条恒等路径直接回传, 避免连乘导致的梯度消失.
//      没有残差的话, 深网络根本训不起来 -- 这是 ResNet 带来的关键结论.
// 所以贯穿全程的那个 n_embd 向量被称为"残差流": 每层往上面加一点东西.
__global__ void k_add(float * x, const float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

static inline int n_blocks(int n, int block) {
    return (n + block - 1) / block;
}

// -----------------------------------------------------------------------------
// cuBLAS 矩阵乘封装
// -----------------------------------------------------------------------------

// y[n_out] = W[n_out, n_in] * x[n_in] (+ bias)
//
// 原理: 线性投影, 也就是 y[o] = sum_i W[o][i] * x[i] + b[o].
//   把 y 的第 o 个分量看成 "W 的第 o 行和 x 做点积", 即用一个学出来的方向
//   去量 x 在该方向上的成分. n_out 行就是 n_out 个不同的探测方向.
//   模型里所有"变换"都是这一个操作: Wq/Wk/Wv 把隐状态投成查询/键/值,
//   Wo 把多头结果融合回去, gate/up/down 做升降维, lm_head 投到词表.
//   算力也几乎全在这里 -- 一次前向的浮点运算 99% 以上是矩阵乘.
//
// 布局对齐是最容易错的地方, 推导一遍:
//   GGUF 里 W 的内存布局是 W[o * n_in + i] (ne0 = n_in 最密).
//   cuBLAS 是列主序, 它把这块内存看成 n_in 行 x n_out 列、lda = n_in 的矩阵 A,
//   即 A[i][o] = W[o * n_in + i].
//   我们要的是 y[o] = sum_i W[o*n_in+i] * x[i] = sum_i A[i][o] * x[i] = (A^T x)[o]
//   所以 A 要转置 (CUBLAS_OP_T), m = n_out, n = 1, k = n_in.
//
// 类型: A/B 都是 F16 (cublasGemmEx 要求两者同类型), C 是 F32, 累加用 F32.
// 单 token 时 n = 1, 其实是 GEMV; prefill 如果改成批量喂 token, n 就变成 token 数.
static void matvec(cublasHandle_t cub, float * y, const __half * W, const __half * x,
                   const float * bias, int n_in, int n_out) {
    const float alpha = 1.0f;
    float       beta  = 0.0f;

    // 有 bias 就先把 bias 拷进 y, 再让 gemm 用 beta=1 累加上去, 省一个 kernel.
    if (bias) {
        CUDA_CHECK(cudaMemcpyAsync(y, bias, sizeof(float) * n_out, cudaMemcpyDeviceToDevice));
        beta = 1.0f;
    }

    CUBLAS_CHECK(cublasGemmEx(cub,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              n_out, 1, n_in,
                              &alpha,
                              W, CUDA_R_16F, n_in,
                              x, CUDA_R_16F, n_in,
                              &beta,
                              y, CUDA_R_32F, n_out,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// -----------------------------------------------------------------------------
// GGUF 元数据读取
// -----------------------------------------------------------------------------

static int64_t kv_find(const gguf_context * gguf, const char * arch, const char * suffix) {
    const std::string key = std::string(arch) + "." + suffix;
    const int64_t     id  = gguf_find_key(gguf, key.c_str());
    return id >= 0 ? id : gguf_find_key(gguf, suffix);
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
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_FLOAT64) {
        return (float) gguf_get_val_f64(gguf, id);
    }
    return gguf_get_val_f32(gguf, id);
}

struct qwen_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;      // 隐藏维度, 一个 token 用多长的向量表示
    uint32_t n_layer;     // Transformer 层数
    uint32_t n_ff;        // FFN 升维后的中间宽度
    uint32_t n_head;      // Q 头数
    uint32_t n_head_kv;   // KV 头数 (GQA, 小于等于 n_head)
    uint32_t n_rot;       // 每个头的维度 = n_embd / n_head, 也是 RoPE 作用的维度
    float    rms_eps;
    float    rope_freq_base;
};

// 显存里的一层权重. 2 维矩阵用 F16 (显存大头), 1 维的 norm/bias 保留 F32 (数值敏感且很小).
struct dev_layer {
    float  * attn_norm = nullptr;
    __half * wq        = nullptr;
    float  * bq        = nullptr;
    __half * wk        = nullptr;
    float  * bk        = nullptr;
    __half * wv        = nullptr;
    float  * bv        = nullptr;
    __half * wo        = nullptr;
    float  * ffn_norm  = nullptr;
    __half * ffn_gate  = nullptr;
    __half * ffn_up    = nullptr;
    __half * ffn_down  = nullptr;
    __half * k_cache   = nullptr; // [n_ctx, n_embd_gqa]
    __half * v_cache   = nullptr;
};

// 跑一次前向需要的全部东西: 超参 + 所有显存指针 + 回读缓冲.
// 打包成一个结构体, 好处是 forward_one_token 只要一个参数就够,
// 而且一眼能看清这个函数依赖哪些状态.
struct qwen_runtime {
    cublasHandle_t cub = nullptr;

    qwen_hparams hp         = {};
    int          n_ctx      = 0;    // KV 槽位数
    int          n_embd_gqa = 0;    // 一组 KV 的宽度 = n_rot * n_head_kv
    int          gqa        = 0;    // 多少个 Q 头共享一组 KV
    float        kq_scale   = 0.0f; // 1 / sqrt(head_dim)

    // 权重, 显存常驻
    __half *               tok_embd    = nullptr;
    float  *               output_norm = nullptr;
    __half *               output      = nullptr; // lm_head, 可能和 tok_embd 是同一块
    std::vector<dev_layer> layers;

    // 中间激活, 每步复用同一块显存
    float  * x      = nullptr; // 残差流 [n_embd]
    __half * xb     = nullptr; // norm 输出, 也是矩阵乘的输入 [n_embd]
    float  * xb2    = nullptr; // 矩阵乘输出, 待加回残差 [n_embd]
    float  * q      = nullptr; // [n_embd]
    float  * k      = nullptr; // [n_embd_gqa]
    float  * v      = nullptr; // [n_embd_gqa]
    float  * att    = nullptr; // 注意力分数暂存 [n_head, n_ctx]
    __half * ao     = nullptr; // 注意力输出 [n_embd]
    float  * gate   = nullptr; // [n_ff]
    float  * up     = nullptr; // [n_ff]
    __half * hb     = nullptr; // silu(gate)*up [n_ff]
    float  * logits = nullptr; // [n_vocab]

    std::vector<float> logits_host; // logits 回读到主机, 在 CPU 上做 argmax
};

// -----------------------------------------------------------------------------
// 前向: 一次跑一个 token
// -----------------------------------------------------------------------------

// 跑一个 token 的完整前向, 返回贪心选出的下一个 token.
//
// 每层是标准的 pre-norm 双子层结构:
//   x = x + Attention(RMSNorm(x))
//   x = x + FFN(RMSNorm(x))
// "pre-norm" 指归一化放在子层入口, 而不是残差相加之后 (post-norm).
// pre-norm 下残差流是一条没有任何归一化的直通路, 深层也能稳定训练;
// 原始 Transformer 论文用的 post-norm 深了之后需要精细调 warmup 才收敛.
//
// n_layer 层重复这套, x 被逐步改写成"下一个词应该是什么"的表示,
// 最后 lm_head 把它投影到词表维度得到 logits.
//
// 所有 kernel 发在默认 stream 上, 按发射顺序串行执行, 所以前后依赖
// (比如先写 KV 再读 KV) 天然满足, 不需要额外同步.
//
// 参数 pos: 本 token 在序列里的绝对位置. 同时决定 RoPE 转多少角度,
//   和 KV cache 写到第几行.
static llama_token forward_one_token(qwen_runtime & rt, llama_token tok, int pos) {
    const qwen_hparams & hp = rt.hp;

    const int n_embd     = (int) hp.n_embd;
    const int n_ff       = (int) hp.n_ff;
    const int n_head     = (int) hp.n_head;
    const int n_head_kv  = (int) hp.n_head_kv;
    const int n_rot      = (int) hp.n_rot;
    const int n_vocab    = (int) hp.n_vocab;
    const int n_embd_gqa = rt.n_embd_gqa;

    // token id -> 初始隐状态
    k_get_row<<<n_blocks(n_embd, 256), 256>>>(rt.x, rt.tok_embd, tok, n_embd);

    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        dev_layer & L = rt.layers[il];

        // --- 子层 1: attention ---
        k_rms_norm<<<1, 256>>>(rt.xb, rt.x, L.attn_norm, n_embd, hp.rms_eps);

        matvec(rt.cub, rt.q, L.wq, rt.xb, L.bq, n_embd, n_embd);
        matvec(rt.cub, rt.k, L.wk, rt.xb, L.bk, n_embd, n_embd_gqa);
        matvec(rt.cub, rt.v, L.wv, rt.xb, L.bv, n_embd, n_embd_gqa);

        k_rope_neox<<<n_blocks(n_head    * n_rot / 2, 128), 128>>>(rt.q, pos, n_rot, n_head,    hp.rope_freq_base);
        k_rope_neox<<<n_blocks(n_head_kv * n_rot / 2, 128), 128>>>(rt.k, pos, n_rot, n_head_kv, hp.rope_freq_base);

        k_store_kv<<<n_blocks(n_embd_gqa, 128), 128>>>(L.k_cache, L.v_cache, rt.k, rt.v, pos, n_embd_gqa);

        k_attn<<<n_head, 128>>>(rt.ao, rt.q, L.k_cache, L.v_cache, rt.att,
                                pos, n_rot, n_head_kv, rt.gqa, rt.n_ctx, rt.kq_scale);

        matvec(rt.cub, rt.xb2, L.wo, rt.ao, nullptr, n_embd, n_embd);
        k_add<<<n_blocks(n_embd, 256), 256>>>(rt.x, rt.xb2, n_embd);

        // --- 子层 2: FFN (SwiGLU) ---
        k_rms_norm<<<1, 256>>>(rt.xb, rt.x, L.ffn_norm, n_embd, hp.rms_eps);

        matvec(rt.cub, rt.gate, L.ffn_gate, rt.xb, nullptr, n_embd, n_ff);
        matvec(rt.cub, rt.up,   L.ffn_up,   rt.xb, nullptr, n_embd, n_ff);

        k_silu_mul<<<n_blocks(n_ff, 256), 256>>>(rt.hb, rt.gate, rt.up, n_ff);

        matvec(rt.cub, rt.xb2, L.ffn_down, rt.hb, nullptr, n_ff, n_embd);
        k_add<<<n_blocks(n_embd, 256), 256>>>(rt.x, rt.xb2, n_embd);
    }

    // --- 最后的 norm + lm_head ---
    // lm_head 把 n_embd 维隐状态投到 n_vocab 维, 每个分量就是对应词的"原始分数"(logit).
    // 这里权重常和 embedding 表共享 (tied): 既然 E 把词映射成向量, 用 E^T
    // 把向量映射回词是自然的对称做法, 还能省下 n_vocab * n_embd 个参数.
    k_rms_norm<<<1, 256>>>(rt.xb, rt.x, rt.output_norm, n_embd, hp.rms_eps);
    matvec(rt.cub, rt.logits, rt.output, rt.xb, nullptr, n_embd, n_vocab);

    CUDA_CHECK(cudaMemcpy(rt.logits_host.data(), rt.logits, sizeof(float) * n_vocab, cudaMemcpyDeviceToHost));
    check_last_kernel("forward");

    // 贪心采样: 直接取 logit 最大的那个词.
    //   完整做法是 softmax(logits / T) 得到概率再按概率抽样, T 是温度.
    //   但 softmax 是单调的, 不改变大小顺序, 所以只要"取最大"就不必真的算 softmax.
    //   贪心等价于温度 T -> 0, 输出完全确定, 便于和 ggml 版本逐字对比验证正确性.
    //   缺点是容易陷入重复; 实际产品会用 temperature + top-k/top-p 引入随机性.
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (rt.logits_host[i] > rt.logits_host[best]) {
            best = i;
        }
    }
    return (llama_token) best;
}

// -----------------------------------------------------------------------------
// 权重加载: GGUF -> 显存
// -----------------------------------------------------------------------------

static size_t g_vram_weights = 0;

// 读一个张量, 统一转成 F16 或 F32, 拷进新 cudaMalloc 的显存.
// 干什么:
//   1) 从 GGUF 目录查形状和类型
//   2) 分块 fread 原始字节 (可能是 F16/F32/Q4_K 任意一种)
//   3) 用 ggml 的 to_float 统一展开成 F32, 需要的话再压成 F16
//   4) cudaMemcpy 上显存
// 为什么分块: token_embd 有上亿个元素, 一次性在内存里展开 F32 要好几个 GB.
// CPU 开销: 磁盘 IO + 类型转换, 是启动耗时的大头. 只跑一次.
static void * load_tensor(gguf_context * gguf, FILE * fp, const char * name, bool as_f16, int64_t * n_elem_out) {
    const int64_t id = gguf_find_tensor(gguf, name);
    if (id < 0) {
        fprintf(stderr, "missing tensor: %s\n", name);
        exit(1);
    }

    const int64_t *  ne   = gguf_get_tensor_ne(gguf, id);
    const ggml_type  type = gguf_get_tensor_type(gguf, id);

    const int64_t n_per_row = ne[0];
    int64_t       n_rows    = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        n_rows *= ne[d];
    }
    const int64_t n_elem    = n_per_row * n_rows;
    const size_t  row_bytes = ggml_row_size(type, n_per_row);
    const size_t  elem_size = as_f16 ? sizeof(__half) : sizeof(float);

    void * dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, (size_t) n_elem * elem_size));
    g_vram_weights += (size_t) n_elem * elem_size;

    // 一次处理约 4M 个元素
    const int64_t chunk = std::max<int64_t>(1, (4 << 20) / n_per_row);

    std::vector<uint8_t>     raw(chunk * row_bytes);
    std::vector<float>       f32(chunk * n_per_row);
    std::vector<ggml_fp16_t> f16(as_f16 ? (size_t) (chunk * n_per_row) : 0);

    // F32 没有 to_float (本来就是 float), 其余类型 (F16/量化) 都靠它展开
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && tr->to_float == nullptr) {
        fprintf(stderr, "tensor %s: type %s cannot be converted\n", name, tr->type_name);
        exit(1);
    }

    const size_t off = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id);
    if (FSEEK(fp, (int64_t) off, SEEK_SET) != 0) {
        die("seek tensor");
    }

    for (int64_t r0 = 0; r0 < n_rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, n_rows - r0);
        const int64_t n  = nr * n_per_row;

        if (fread(raw.data(), 1, (size_t) nr * row_bytes, fp) != (size_t) nr * row_bytes) {
            die("read tensor");
        }

        const float * src;
        if (type == GGML_TYPE_F32) {
            src = (const float *) raw.data();
        } else {
            tr->to_float(raw.data(), f32.data(), n);
            src = f32.data();
        }

        const void * host;
        if (as_f16) {
            ggml_fp32_to_fp16_row(src, f16.data(), n);
            host = f16.data();
        } else {
            host = src;
        }

        CUDA_CHECK(cudaMemcpy((char *) dev + (size_t) r0 * n_per_row * elem_size, host,
                              (size_t) n * elem_size, cudaMemcpyHostToDevice));
    }

    if (n_elem_out) {
        *n_elem_out = n_elem;
    }
    return dev;
}

// -----------------------------------------------------------------------------
// 反分词 (detokenize): token id -> 文本
//
// 模型吐出来的是词表下标, 要变成屏幕上的字就得查词表.
// 这里不调 llama_token_to_piece, 直接从 GGUF 的 tokenizer.ggml.* 自己读自己解.
//
// 难点是 Qwen2 用的 GPT-2 byte-level BPE (tokenizer.ggml.model = gpt2).
// 为了让词表里不出现控制字符和非法 UTF-8 序列, 它把原始字节重映射成可打印字符再存:
//   0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF 这些字节, 映射成同码点的字符 (看着没变)
//   剩下 68 个字节 (0x00-0x20, 0x7F-0xA0, 0xAD), 依次映射到码点 256, 257, ...
// 所以空格 0x20 在词表里存的是 "Ġ" (它是这 68 个里的第 33 个, 256+32 = U+0120),
// 换行 0x0A 存的是 "Ċ" (256+10 = U+010A). 不还原的话满屏都是 Ġ 和 Ċ.
//
// 反方向 (文本 -> token) 没有自己写: 那要 BPE merges 合并表, 加上 GPT-2 的正则
// 预切分, 后者依赖完整的 Unicode 字符属性表 (llama.cpp 里光这张表就几千行).
// 那部分是纯文本处理, 和推理数学无关, 继续用 llama 的分词器.
// -----------------------------------------------------------------------------

// GGUF tokenizer.ggml.token_type 的取值.
// Qwen2.5-0.5B 只出现 1/3/4/5 这四种, 没有 BYTE 类型, 所以不需要 <0xXX> 分支.
enum token_type {
    TOKEN_TYPE_NORMAL       = 1,
    TOKEN_TYPE_UNKNOWN      = 2,
    TOKEN_TYPE_CONTROL      = 3, // <|im_end|> <|endoftext|> 这种
    TOKEN_TYPE_USER_DEFINED = 4,
    TOKEN_TYPE_UNUSED       = 5,
};

// 码点反查表的大小: 原样映射最大到 255, 额外映射最大到 256 + 67 = 323.
#define CPT_MAP_SIZE 324

struct qwen_vocab {
    std::vector<std::string> text; // 词表原文, 还是字节映射后的编码态
    std::vector<int32_t>     type;
    int16_t cpt_to_byte[CPT_MAP_SIZE]; // 码点 -> 原始字节, -1 表示该码点没被用到
};

// 判断一个字节在 byte-level BPE 里是否"原样保留"
static bool bpe_byte_is_printable(int ch) {
    return (ch >= 0x21 && ch <= 0x7E) || (ch >= 0xA1 && ch <= 0xAC) || (ch >= 0xAE && ch <= 0xFF);
}

// 从 GGUF 读词表并建好反查表. 启动时跑一次, 15 万个字符串, 几十毫秒.
static void load_vocab(const gguf_context * gguf, qwen_vocab & v) {
    const int64_t id_tok = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (id_tok < 0) {
        die("gguf has no tokenizer.ggml.tokens");
    }
    const size_t n = gguf_get_arr_n(gguf, id_tok);
    v.text.resize(n);
    for (size_t i = 0; i < n; ++i) {
        v.text[i] = gguf_get_arr_str(gguf, id_tok, i);
    }

    // 没有 token_type 就全当普通 token
    v.type.assign(n, TOKEN_TYPE_NORMAL);
    const int64_t id_type = gguf_find_key(gguf, "tokenizer.ggml.token_type");
    if (id_type >= 0 && gguf_get_arr_n(gguf, id_type) == n) {
        const int32_t * p = (const int32_t *) gguf_get_arr_data(gguf, id_type);
        for (size_t i = 0; i < n; ++i) {
            v.type[i] = p[i];
        }
    }

    // 建码点 -> 字节的反查表, 规则和文件头注释里的映射一一对应
    for (int i = 0; i < CPT_MAP_SIZE; ++i) {
        v.cpt_to_byte[i] = -1;
    }
    for (int ch = 0; ch < 256; ++ch) {
        if (bpe_byte_is_printable(ch)) {
            v.cpt_to_byte[ch] = (int16_t) ch;
        }
    }
    int extra = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (!bpe_byte_is_printable(ch)) {
            v.cpt_to_byte[256 + extra] = (int16_t) ch;
            ++extra;
        }
    }
}

// token id -> 它代表的原始字节. 每生成一个 token 调一次, 纯 CPU, 开销可忽略.
static std::string detokenize(const qwen_vocab & v, llama_token tok) {
    if (tok < 0 || (size_t) tok >= v.text.size()) {
        return "";
    }
    const std::string & s = v.text[tok];

    // 控制符/用户自定义 token 本身就是 <|im_end|> 这种可读字面量, 没走字节映射
    if (v.type[tok] != TOKEN_TYPE_NORMAL) {
        return s;
    }

    // 普通 token: 先按 UTF-8 解出码点, 再逐个映射回原始字节
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = (uint8_t) s[i];

        uint32_t cpt;
        size_t   len;
        if (c < 0x80) {
            cpt = c;        len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cpt = c & 0x1F; len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cpt = c & 0x0F; len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cpt = c & 0x07; len = 4;
        } else {
            ++i; continue; // 非法起始字节, 跳过
        }
        if (i + len > s.size()) {
            break;
        }
        for (size_t k = 1; k < len; ++k) { // 后续字节各贡献低 6 位
            cpt = (cpt << 6) | ((uint8_t) s[i + k] & 0x3F);
        }
        i += len;

        if (cpt < CPT_MAP_SIZE && v.cpt_to_byte[cpt] >= 0) {
            out += (char) v.cpt_to_byte[cpt];
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// 聊天模板
// -----------------------------------------------------------------------------

// 用 GGUF 自带的模板把原始问题包成模型认识的对话格式.
// 模板套错 (比如给 R1 蒸馏套 ChatML) 模型会自言自语而不是回答.
static std::string apply_chat_prompt(const llama_model * model, const std::string & user) {
    if (user.find("<|im_start|>") != std::string::npos) {
        return user; // 用户已经自己写好了
    }
    llama_chat_message msg  = { "user", user.c_str() };
    const char *       tmpl = llama_model_chat_template(model, nullptr);

    std::vector<char> buf(user.size() * 4 + 256);
    int32_t           n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    if (n > (int32_t) buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
    }
    if (n <= 0) {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    }
    return std::string(buf.data(), (size_t) n);
}

static void print_usage(const char * argv0) {
    printf("usage: %s -m model.gguf [-p prompt] [-n n_predict] [-c n_ctx]\n", argv0);
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    ggml_log_set(quiet_log, nullptr);
    llama_log_set(quiet_log, nullptr);

    std::string path;
    std::string prompt    = "who are you?";
    int         n_predict = 64;
    int         n_ctx     = 512; // KV 槽位数, 决定 KV cache 显存

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
        print_usage(argv[0]);
        return 1;
    }

    // =========================================================================
    // 1) 分词. 唯一还在用 llama 模型接口的地方.
    //    vocab_only=true: 只解析 tokenizer.*, 不加载权重.
    // =========================================================================
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = true;
    mp.n_gpu_layers = 0;

    llama_model * vocab_model = llama_model_load_from_file(path.c_str(), mp);
    if (!vocab_model) {
        die("vocab_only load failed");
    }
    const llama_vocab * vocab   = llama_model_get_vocab(vocab_model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    prompt = apply_chat_prompt(vocab_model, prompt);

    const int n_tok = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, add_bos, true);
    if (n_tok <= 0) {
        die("tokenize size");
    }
    std::vector<llama_token> prompt_toks(n_tok);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_toks.data(), n_tok, add_bos, true) < 0) {
        die("tokenize");
    }
    if (n_tok + n_predict > n_ctx) {
        die("n_ctx too small for prompt + n_predict");
    }

    // =========================================================================
    // 2) CUDA 初始化. 这里不用 ggml backend, 直接建 CUDA context + cuBLAS handle.
    // =========================================================================
    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev == 0) {
        die("no CUDA device found (this example requires CUDA)");
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("|---Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    cublasHandle_t cub = nullptr;
    CUBLAS_CHECK(cublasCreate(&cub));
    print_vram("|| after cuda init");

    // =========================================================================
    // 3) 读超参
    // =========================================================================
    gguf_init_params gp   = { true, nullptr }; // no_alloc: 只读目录, 不映射权重
    gguf_context *   gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) {
        die("gguf open");
    }
    // 在这里打印 gguf 中的内容的描述
    // 先打印所有 KV 元数据 (超参 / 词表指针 / 训练信息等), 再打印张量目录.
    {
        const int64_t n_kv = gguf_get_n_kv(gguf);
        printf("|---GGUF metadata: %lld kv pairs\n", (long long) n_kv);
        for (int64_t i = 0; i < n_kv; ++i) {
            const char *    key = gguf_get_key(gguf, i);
            const enum gguf_type t = gguf_get_kv_type(gguf, i);
            printf("|   [%2lld] %-40s : ", (long long) i, key);
            switch (t) {
                case GGUF_TYPE_UINT8:   printf("u8  = %u",   (unsigned) gguf_get_val_u8(gguf, i)); break;
                case GGUF_TYPE_INT8:    printf("i8  = %d",   (int) gguf_get_val_i8(gguf, i)); break;
                case GGUF_TYPE_UINT16:  printf("u16 = %u",   (unsigned) gguf_get_val_u16(gguf, i)); break;
                case GGUF_TYPE_INT16:   printf("i16 = %d",   (int) gguf_get_val_i16(gguf, i)); break;
                case GGUF_TYPE_UINT32:  printf("u32 = %u",   gguf_get_val_u32(gguf, i)); break;
                case GGUF_TYPE_INT32:   printf("i32 = %d",   gguf_get_val_i32(gguf, i)); break;
                case GGUF_TYPE_FLOAT32: printf("f32 = %g",   gguf_get_val_f32(gguf, i)); break;
                case GGUF_TYPE_UINT64:  printf("u64 = %llu", (unsigned long long) gguf_get_val_u64(gguf, i)); break;
                case GGUF_TYPE_INT64:   printf("i64 = %lld", (long long) gguf_get_val_i64(gguf, i)); break;
                case GGUF_TYPE_FLOAT64: printf("f64 = %g",   gguf_get_val_f64(gguf, i)); break;
                case GGUF_TYPE_BOOL:    printf("bool = %s",  gguf_get_val_bool(gguf, i) ? "true" : "false"); break;
                case GGUF_TYPE_STRING:  printf("str = \"%s\"", gguf_get_val_str(gguf, i)); break;
                case GGUF_TYPE_ARRAY: {
                    const enum gguf_type at = gguf_get_arr_type(gguf, i);
                    const int64_t        an = gguf_get_arr_n(gguf, i);
                    printf("arr[%lld] type=%d", (long long) an, (int) at);
                    break;
                }
                default: printf("type=%d", (int) t); break;
            }
            printf("\n");
        }

        const int64_t n_t = gguf_get_n_tensors(gguf);
        printf("|---GGUF tensors: %lld tensors\n", (long long) n_t);
        for (int64_t i = 0; i < n_t; ++i) {
            const char *   name = gguf_get_tensor_name(gguf, i);
            const ggml_type type = gguf_get_tensor_type(gguf, i);
            const int64_t * ne  = gguf_get_tensor_ne(gguf, i);
            printf("|   [%3lld] %-40s type=%-12s shape=[", (long long) i, name,
                   ggml_type_name(type));
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                printf("%s%lld", d ? ", " : "", (long long) ne[d]);
            }
            // 估计元素个数 (形状各维相乘), 不依赖额外 API
            int64_t n_elem = 1;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                n_elem *= ne[d];
            }
            printf("] elems=%lld (%.2f MB)\n", (long long) n_elem,
                   (double) n_elem * ggml_type_size(type) / (1024.0 * 1024.0));
        }
    }

    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id < 0) {
        die("no general.architecture");
    }
    const char * arch = gguf_get_val_str(gguf, arch_id);
    if (strcmp(arch, "qwen2") != 0) {
        fprintf(stderr, "this example only implements qwen2 (got %s)\n", arch);
        return 1;
    }

    qwen_hparams hp = {};
    hp.n_embd  = kv_u32(gguf, arch, "embedding_length");
    hp.n_layer = kv_u32(gguf, arch, "block_count");
    hp.n_ff    = kv_u32(gguf, arch, "feed_forward_length");
    hp.n_head  = kv_u32(gguf, arch, "attention.head_count");
    {
        const int64_t id = kv_find(gguf, arch, "attention.head_count_kv");
        hp.n_head_kv = id >= 0 ? gguf_get_val_u32(gguf, id) : hp.n_head;
    }
    hp.rms_eps        = kv_f32_or(gguf, arch, "attention.layer_norm_rms_epsilon", 1e-6f);
    hp.rope_freq_base = kv_f32_or(gguf, arch, "rope.freq_base", 10000.0f);
    hp.n_rot          = hp.n_embd / hp.n_head;

    // 词表也在 GGUF 里, 趁 gguf 还没 free 读出来, 后面输出时把 token id 还原成文本
    qwen_vocab vocab_gguf;
    load_vocab(gguf, vocab_gguf);

    const int n_embd     = (int) hp.n_embd;
    const int n_ff       = (int) hp.n_ff;
    const int n_head     = (int) hp.n_head;
    const int n_head_kv  = (int) hp.n_head_kv;
    const int n_rot      = (int) hp.n_rot;
    const int n_embd_gqa = n_rot * n_head_kv;
    const int gqa        = n_head / n_head_kv;

    if (n_head % n_head_kv != 0) {
        die("n_head must be a multiple of n_head_kv");
    }
    if (n_rot % 2 != 0) {
        die("head dim must be even for RoPE");
    }

    // =========================================================================
    // 4) 权重上显存. 先估算够不够, 不够就早点报错, 别等 cudaMalloc 失败.
    // =========================================================================
    const bool   tied      = gguf_find_tensor(gguf, "output.weight") < 0;
    const int64_t n_vocab_t = gguf_get_tensor_ne(gguf, gguf_find_tensor(gguf, "token_embd.weight"))[1];
    hp.n_vocab = (uint32_t) n_vocab_t;

    printf("qwen2: n_layer=%u n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_rot=%d rope_base=%.0f n_vocab=%u%s\n",
           hp.n_layer, n_embd, n_head, n_head_kv, n_ff, n_rot, hp.rope_freq_base, hp.n_vocab,
           tied ? " (tied lm_head)" : "");

    {
        const size_t w_embd  = (size_t) n_vocab_t * n_embd * 2 * (tied ? 1 : 2);
        const size_t w_layer = ((size_t) 2 * n_embd * n_embd + 2 * (size_t) n_embd_gqa * n_embd +
                                3 * (size_t) n_ff * n_embd) * 2;
        const size_t kv      = (size_t) hp.n_layer * n_ctx * n_embd_gqa * 2 * 2;
        const size_t need    = w_embd + w_layer * hp.n_layer + kv;

        size_t free_b  = 0;
        size_t total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        printf("need about %s of vram (weights f16 + kv cache), free %s\n",
               human_size((double) need).c_str(), human_size((double) free_b).c_str());
        if (need > free_b) {
            fprintf(stderr,
                    "error: not enough vram. use a smaller model (e.g. Qwen2.5-0.5B-Instruct-fp16)\n"
                    "       or lower -c (current %d)\n", n_ctx);
            return 1;
        }
    }

    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        die("open gguf for weights");
    }

    // 从这里开始, 所有显存指针都挂在 rt 上, 前向时整个结构体传给 forward_one_token
    qwen_runtime rt;
    rt.cub        = cub;
    rt.hp         = hp;
    rt.n_ctx      = n_ctx;
    rt.n_embd_gqa = n_embd_gqa;
    rt.gqa        = gqa;
    rt.kq_scale   = 1.0f / sqrtf((float) n_rot);

    printf("uploading weights...\n");
    rt.tok_embd    = (__half *) load_tensor(gguf, fp, "token_embd.weight", true, nullptr);
    rt.output_norm = (float *)  load_tensor(gguf, fp, "output_norm.weight", false, nullptr);
    rt.output      = tied ? rt.tok_embd : (__half *) load_tensor(gguf, fp, "output.weight", true, nullptr);

    rt.layers.resize(hp.n_layer);
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        dev_layer & L = rt.layers[il];
        char        name[128];

#define LOAD(field, fmt, as_f16)                                              \
        snprintf(name, sizeof(name), fmt, il);                                \
        L.field = (decltype(L.field)) load_tensor(gguf, fp, name, as_f16, nullptr)

        LOAD(attn_norm, "blk.%u.attn_norm.weight",   false);
        LOAD(wq,        "blk.%u.attn_q.weight",      true);
        LOAD(bq,        "blk.%u.attn_q.bias",        false);
        LOAD(wk,        "blk.%u.attn_k.weight",      true);
        LOAD(bk,        "blk.%u.attn_k.bias",        false);
        LOAD(wv,        "blk.%u.attn_v.weight",      true);
        LOAD(bv,        "blk.%u.attn_v.bias",        false);
        LOAD(wo,        "blk.%u.attn_output.weight", true);
        LOAD(ffn_norm,  "blk.%u.ffn_norm.weight",    false);
        LOAD(ffn_gate,  "blk.%u.ffn_gate.weight",    true);
        LOAD(ffn_up,    "blk.%u.ffn_up.weight",      true);
        LOAD(ffn_down,  "blk.%u.ffn_down.weight",    true);
#undef LOAD

        // KV cache 不在 GGUF 里, 运行时开. 清零: 没写过的槽不会被读, 但别留显存垃圾.
        const size_t kv_bytes = (size_t) n_ctx * n_embd_gqa * sizeof(__half);
        CUDA_CHECK(cudaMalloc(&L.k_cache, kv_bytes));
        CUDA_CHECK(cudaMalloc(&L.v_cache, kv_bytes));
        CUDA_CHECK(cudaMemset(L.k_cache, 0, kv_bytes));
        CUDA_CHECK(cudaMemset(L.v_cache, 0, kv_bytes));

        printf("  layer %u / %u, kv_bytes: %d", il + 1, hp.n_layer, kv_bytes);
        fflush(stdout);
    }
    printf("\n");
    fclose(fp);
    gguf_free(gguf);

    printf("weights on device: %s\n", human_size((double) g_vram_weights).c_str());
    print_vram("after weights");

    // =========================================================================
    // 5) 中间激活的显存. 每步复用同一块, 不反复申请.
    //    残差流和归约相关的量用 F32, 要喂给 cuBLAS 的用 F16.
    // =========================================================================
    CUDA_CHECK(cudaMalloc(&rt.x,      sizeof(float)  * n_embd));
    CUDA_CHECK(cudaMalloc(&rt.xb,     sizeof(__half) * n_embd));
    CUDA_CHECK(cudaMalloc(&rt.xb2,    sizeof(float)  * n_embd));
    CUDA_CHECK(cudaMalloc(&rt.q,      sizeof(float)  * n_embd));
    CUDA_CHECK(cudaMalloc(&rt.k,      sizeof(float)  * n_embd_gqa));
    CUDA_CHECK(cudaMalloc(&rt.v,      sizeof(float)  * n_embd_gqa));
    CUDA_CHECK(cudaMalloc(&rt.att,    sizeof(float)  * n_head * n_ctx));
    CUDA_CHECK(cudaMalloc(&rt.ao,     sizeof(__half) * n_embd));
    CUDA_CHECK(cudaMalloc(&rt.gate,   sizeof(float)  * n_ff));
    CUDA_CHECK(cudaMalloc(&rt.up,     sizeof(float)  * n_ff));
    CUDA_CHECK(cudaMalloc(&rt.hb,     sizeof(__half) * n_ff));
    CUDA_CHECK(cudaMalloc(&rt.logits, sizeof(float)  * hp.n_vocab));

    rt.logits_host.resize(hp.n_vocab);
    print_vram("after activations");

    // =========================================================================
    // 6) 主循环: prefill 把 prompt 逐 token 灌进 KV cache, 然后自回归生成.
    //    每个 token 的前向都在 forward_one_token 里, 见文件上半部分.
    // =========================================================================
    printf("<<<PROMPT_START>>>\n%s\n<<<PROMPT_END>>>\n", prompt.c_str());
    fflush(stdout);

    int         pos  = 0; // 下一个 token 在序列里的绝对位置, 两个阶段连续累加
    llama_token last = 0; // 上一次前向预测出的 token

    // --- 阶段 1: prefill (预填充) ---
    // 输入是已知的 prompt token, 目的只是把它们的 K/V 写进 cache.
    // 每步的输出 logits 都丢掉不看, 唯独最后一步例外:
    // 那一步吃的是 prompt 最后一个 token, 它预测出的就是回复的第一个词.
    // 注意这里是逐 token 过的, 真实引擎会把整段 prompt 一次性批量喂进去 (见下面说明).
    for (; pos < n_tok; ++pos) {
        last = forward_one_token(rt, prompt_toks[pos], pos);
    }

    printf("<<<OUTPUT_START>>>\n");
    fflush(stdout);

    // --- 阶段 2: decode (解码/自回归生成) ---
    // 输入不再来自 prompt, 而是上一步自己吐出来的 token: 输出接回输入, 循环下去.
    // 每步产出 1 个新词, 所以生成 N 个词就要跑 N 次完整前向.
    // 退出条件: 撞到结束符 (eos), 或达到 -n 上限, 或 KV cache 写满.
    std::string response;
    int         n_gen   = 0;
    bool        hit_eog = false;
    for (; n_gen < n_predict && pos < n_ctx; ++n_gen, ++pos) {
        if (llama_vocab_is_eog(vocab, last)) {
            hit_eog = true;
            break;
        }
        const std::string piece = detokenize(vocab_gguf, last);
        if (!piece.empty()) {
            fwrite(piece.data(), 1, piece.size(), stdout);
            fflush(stdout);
            response += piece;
        }
        last = forward_one_token(rt, last, pos);
    }

    printf("\n<<<OUTPUT_END>>> tokens=%d stop=%s\n", n_gen, hit_eog ? "eos" : "n_predict");
    printf("<<<RESULT_START>>>\n%s\n<<<RESULT_END>>>\n", response.c_str());
    fflush(stdout);

    // =========================================================================
    // 7) 收尾. 进程退出时驱动本来就会回收显存, 这里显式释放只是为了
    //    让读者看清前面申请过哪些资源.
    // =========================================================================
    for (auto & L : rt.layers) {
        cudaFree(L.attn_norm);
        cudaFree(L.wq); cudaFree(L.bq);
        cudaFree(L.wk); cudaFree(L.bk);
        cudaFree(L.wv); cudaFree(L.bv);
        cudaFree(L.wo);
        cudaFree(L.ffn_norm);
        cudaFree(L.ffn_gate);
        cudaFree(L.ffn_up);
        cudaFree(L.ffn_down);
        cudaFree(L.k_cache);
        cudaFree(L.v_cache);
    }
    cudaFree(rt.tok_embd);
    cudaFree(rt.output_norm);
    if (!tied) {
        cudaFree(rt.output);
    }
    cudaFree(rt.x);   cudaFree(rt.xb); cudaFree(rt.xb2);
    cudaFree(rt.q);   cudaFree(rt.k);  cudaFree(rt.v);
    cudaFree(rt.att); cudaFree(rt.ao);
    cudaFree(rt.gate); cudaFree(rt.up); cudaFree(rt.hb);
    cudaFree(rt.logits);

    cublasDestroy(cub);
    llama_model_free(vocab_model);
    llama_backend_free();
    return 0;
}
