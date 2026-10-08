/**
 * simple-cuda.cpp
 * ----------------------------------------------------------------------------
 * 一个“最小可运行”的示例，演示如何绕过 llama 模型，直接调用 ggml 的 CUDA 后端
 * 去做一次真实的计算（矩阵乘 C = A·B，以及逐元素加 D = C + bias）。
 *
 * 这正好对应我们前面聊的调用链：
 *   ggml_backend_dev_init()        -> 拿到一块 GPU 的 backend 句柄
 *   ggml_backend_alloc_ctx_tensors -> 在显存里为 tensor 分配空间
 *   ggml_backend_tensor_set        -> 把 host 数据拷进显存（cudaMemcpyAsync 的封装）
 *   ggml_backend_graph_compute     -> 真正发射 CUDA kernel 去算
 *   ggml_backend_tensor_get        -> 把显存结果拷回 host
 *
 * 根目录下执行:
 *   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
 *   cmake --build build-cuda --target simple-cuda
 *   ./build-cuda/bin/simple-cuda
 *
 * 注意：GGML_CUDA 默认是 OFF。如果没用 -DGGML_CUDA=ON 编译，示例会自动
 *       退化到 CPU backend，逻辑完全一样，只是计算发生在 CPU 上。
 * ----------------------------------------------------------------------------
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// 把字节数格式化成人能读的字符串（B/KB/MB/GB）
static std::string human_size(double bytes) {
    const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    int u = 0;
    double v = bytes;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return std::string(buf);
}

int main(void) {
    // 矩阵尺寸：A 是 (M x K)，B 是 (K x N)，结果 C 是 (M x N)
    const int64_t M = 64;
    const int64_t K = 128;
    const int64_t N = 96;

    // ===== 步骤 0：注册所有后端（CPU / CUDA / Metal ...）=====
    // 只有这一步之后，ggml_backend_dev_by_type(GPU) 才能找到 CUDA 设备。
    ggml_backend_load_all();

    // ===== 步骤 1：选设备 + 创建 backend =====
    // 优先用独显(GPU)，没有就退化为集显(iGPU)，再没有就用 CPU。
    ggml_backend_dev_t dev =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

    if (!dev) {
        fprintf(stderr, "没有找到任何可用的计算设备\n");
        return 1;
    }

    // 打印选中的设备信息
    struct ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    const bool is_gpu = (props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                         props.type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    printf("使用设备        : %s (%s)\n", props.name, is_gpu ? "GPU/CUDA" : "CPU");
    printf("设备描述        : %s\n", props.description ? props.description : "");
    printf("显存            : 共 %s，空闲 %s\n",
           human_size((double) props.memory_total).c_str(),
           human_size((double) props.memory_free).c_str());

    // ggml_backend_dev_init 内部会做：选 CUDA 卡 + 建 CUDA context + 建 stream
    // 返回的 ggml_backend_t 就是后续所有计算的“执行环境”。
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        fprintf(stderr, "backend 初始化失败\n");
        return 1;
    }

    // ===== 步骤 2：用 ggml 搭建计算图 =====
    // ggml_init 只负责“记账”（tensor 的元数据），no_alloc=true 表示先不分配显存，
    // 显存稍后由 ggml_backend_alloc_ctx_tensors 统一在 GPU 上分配。
    static size_t buf_meta[GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead()];
    struct ggml_init_params params = {
        /*.mem_size   =*/ sizeof(buf_meta),
        /*.mem_buffer =*/ buf_meta,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 输入 A、B、bias 在数据搬运时是“源”，标记为 input
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // A^T 布局，便于做 mul_mat
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
    struct ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_set_input(bias);

    // 核心计算：C = A · B（矩阵乘，是整个大模型里最占算力的算子，对应 CUDA 上的 matmul kernel）
    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    // 再加偏置：D = C + bias（逐元素加，对应 CUDA 上的 elementwise add kernel）
    struct ggml_tensor * d = ggml_add(ctx, c, bias);
    ggml_set_output(d); // 只有标记 output 的结果会被保留，不会被图分配器提前覆盖

    // 把根节点展开成一张可执行的 ggml_cgraph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, d);

    // ===== 步骤 3：在 GPU 显存里为整张图分配空间 =====
    // 这一行就是“把 tensor 真正放进显存”，之后 tensor->data 指向的是 GPU 显存地址。
    if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) {
        fprintf(stderr, "显存分配失败（可能 GPU 显存不足）\n");
        return 1;
    }

    // ===== 步骤 4：准备 host 侧数据，并拷入显存 =====
    std::vector<float> h_a(M * K), h_b(K * N), h_bias(N);
    for (int64_t i = 0; i < M * K; i++) h_a[i] = (float)((i % 7) - 3);   // 随便填些数
    for (int64_t i = 0; i < K * N; i++) h_b[i] = (float)((i % 5) - 2);
    for (int64_t i = 0; i < N;     i++) h_bias[i] = 1.0f;

    // ggml_backend_tensor_set 内部调用 cudaMemcpyAsync 把 host 数据上传到显存
    ggml_backend_tensor_set(a,    h_a.data(),    0, ggml_nbytes(a));
    ggml_backend_tensor_set(b,    h_b.data(),    0, ggml_nbytes(b));
    ggml_backend_tensor_set(bias, h_bias.data(), 0, ggml_nbytes(bias));

    printf("\n开始计算（调用 ggml_backend_graph_compute 发射 CUDA kernel）...\n");

    // ===== 步骤 5：真正触发 GPU 计算 =====
    // 这一步会沿着计算图逐个节点调用 ggml_cuda_compute_forward，
    // 对 mul_mat 节点发射 matmul kernel，对 add 节点发射 elementwise kernel。
    enum ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "图计算失败，status=%d\n", (int) st);
        return 1;
    }

    // ===== 步骤 6：把结果从显存拷回 host =====
    std::vector<float> h_c(M * N), h_d(M * N);
    ggml_backend_tensor_get(c, h_c.data(), 0, ggml_nbytes(c));
    ggml_backend_tensor_get(d, h_d.data(), 0, ggml_nbytes(d));

    // ===== 步骤 7：用 CPU 重算一遍做校验 =====
    // C = A·B，注意 a 是 (K x M) 布局，b 是 (N x K) 布局
    double max_err_c = 0, max_err_d = 0;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                // a 的 (k, i)，b 的 (j, k)
                acc += h_a[k * M + i] * h_b[j * K + k];
            }
            max_err_c = std::max(max_err_c, (double)std::fabs(acc - h_c[i * N + j]));
            max_err_d = std::max(max_err_d, (double)std::fabs(acc + h_bias[j] - h_d[i * N + j]));
        }
    }

    printf("矩阵乘 C=A·B 最大误差 : %.6e\n", max_err_c);
    printf("加偏置 D=C+bias 最大误差: %.6e\n", max_err_d);

    // 顺带打印几个结果样本，确认不是全 0
    printf("\n结果样本（前 5 个 D 值）: ");
    for (int i = 0; i < 5 && i < M * N; i++) printf("%.3f ", h_d[i]);
    printf("\n");

    // ===== 清理 =====
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (max_err_c < 1e-3 && max_err_d < 1e-3) {
        printf("\n校验通过：GPU 计算结果与 CPU 参考一致 ✅\n");
        return 0;
    }
    printf("\n校验失败：误差过大 ❌\n");
    return 1;
}
