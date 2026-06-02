---
title: 基于 LD_PRELOAD 的 CUDA Runtime 拦截机制
published: 2026-05-27T16:15:59.465Z
description: ""
updated: ""
tags:
  - CUDA
category: CUDA
draft: false
pin: 0
toc: true
lang: ""
abbrlink: cuda-interception
---
在一些场景下，我们不希望执行实际的 CUDA Kernel，只希望观察或替代 host 侧对 CUDA API 的调用。这可以通过 `LD_PRELOAD` 让动态链接器优先加载自定义 `.so` 中的同名符号，从而拦截 CUDA Runtime API，例如 `cudaLaunchKernel`，再选择转发给真实实现或直接返回。

除此之外，如果要完整 GPU 虚拟化，不仅要拦截所有 host 到 GPU 的 API 调用，还必须维护一个与真实 GPU 一致的虚拟设备状态，并正确模拟 kernel、memory、stream、event、communication 等 side effects。
1. GPU memory state：`cudaMalloc` 返回的地址、device memory 内容、KV cache、tensor buffer、allocator 行为。
2. Execution semantics：kernel 执行后的结果、stream 顺序、event、同步、依赖关系。
3. Kernel side effects：kernel 对 global memory、shared memory、atomic、random state、NCCL buffer 的修改。
4. Driver/runtime/library 层：不只 `cudaLaunchKernel`，还有 `cuLaunchKernel`、cuBLAS、cuDNN、NCCL、CUDA Graph、Triton runtime、custom extension。
5. 多 GPU 语义：rank、device ordinal、peer access、collective communication、topology、failure、barrier、overlap。

## 参考代码

下面代码是实现拦截 `cudaLaunchKernel` 的最小 API[^1]。

[^1]: 要想完全拦截 GPU 的其他操作，至少还需要实现 `cudaMalloc / cudaFree / cudaMallocAsync`, `cudaMemcpy / cudaMemcpyAsync`, `cudaMemset / cudaMemsetAsync`, `cudaStreamCreate / cudaStreamSynchronize`, `cudaEventRecord / cudaEventSynchronize / cudaEventElapsedTime`, `cudaGraphLaunch / cudaGraphInstantiate`, `cuLaunchKernel    // Driver API，非常重要`,  `NCCL calls        // ncclAllReduce / ncclSend / ncclRecv ...` 等其他层

```cpp
// cuda_intercept.cpp
#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>

using cudaLaunchKernel_t = cudaError_t (*)(
    const void*, dim3, dim3, void**, size_t, cudaStream_t);

using cudaMemcpyAsync_t = cudaError_t (*)(
    void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);

extern "C" cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream) {

    static auto real_cudaLaunchKernel =
        (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");

    fprintf(stderr,
            "[CUDA LAUNCH] func=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu stream=%p\n",
            func,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            sharedMem,
            stream);

    // 方案 A：真实执行
    return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    // 方案 B：假执行，不真的跑 kernel
    // return cudaSuccess;
}

extern "C" cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {

    static auto real_cudaMemcpyAsync =
        (cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    fprintf(stderr,
            "[CUDA MEMCPY ASYNC] bytes=%zu kind=%d stream=%p\n",
            count,
            (int)kind,
            stream);

    return real_cudaMemcpyAsync(dst, src, count, kind, stream);

    // 假执行：
    // return cudaSuccess;
}
```

要使用这个代码，例如我们希望执行矩阵乘法：

```cpp
# test.py
import torch

x = torch.randn(4096, 4096, device="cuda")
y = torch.randn(4096, 4096, device="cuda")

z = x @ y

torch.cuda.synchronize()

print(z.shape)
```

首先先编译我们的 C++ 代码，编译成 `libcuda_intercept.so` 文件

```bash
g++ -shared -fPIC cuda_intercept.cpp -o libcuda_intercept.so \
    -ldl -I/usr/local/cuda/include
```

然后在实际运行的时候通过 `LD_PRELOAD` 把 `cudaLaunchKernel` 进行截胡：

```bash
LD_PRELOAD=$PWD/libcuda_intercept.so python test.py
```

代码中 `real_cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync")` 可以运行把这个函数传递下去，去调用真正的 runtime.
## 参考资料

- [Singularity: Planet-Scale, Preemptive and Elastic Scheduling of AI Workloads](https://arxiv.org/abs/2202.07848)