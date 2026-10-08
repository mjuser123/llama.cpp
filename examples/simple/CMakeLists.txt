set(TARGET llama-simple)
add_executable(${TARGET} simple.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-simple-prompt)
add_executable(${TARGET} simple-prompt.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 直接调用 ggml CUDA 后端做计算的最小示例（不依赖 llama 模型）
# 编译时务必加 -DGGML_CUDA=ON 才能用到真实 CUDA 内核；否则自动退化到 CPU backend
set(TARGET simple-cuda)
add_executable(${TARGET} simple-cuda.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Parse GGUF metadata and print what each part is / where it goes
set(TARGET gguf-inspect)
add_executable(${TARGET} gguf-inspect.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Qwen2 CUDA infer: llama tokenizer only, then ggml CUDA graph (no llama_decode)
set(TARGET simple-qwen2-cuda)
add_executable(${TARGET} simple-qwen2-cuda.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Qwen2 raw CUDA infer: hand-written kernels + cuBLAS, no ggml ops and no ggml graph
# needs -DGGML_CUDA=ON; model is converted to f16 on load, so it must fit in vram
if (GGML_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(TARGET simple-qwen2-cuda-raw)
    add_executable(${TARGET} simple-qwen2-cuda-raw.cu)
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE llama CUDA::cudart CUDA::cublas ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # ggml-cuda picks the architectures in its own directory scope, so set them again here
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES native)
    endif()
endif()
