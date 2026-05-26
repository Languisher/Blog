---
title: CUDA Grid, Block, Thread 和寻址
published: 2026-03-04T14:45:48.564Z
description: 本文从数据并行计算的角度出发，介绍 CUDA 中 Grid、Block 和 Thread 的分层组织方式，并解释如何通过线程索引将计算任务映射到具体数据。
updated: 2026-04-26T10:18:01Z
tags:
  - CUDA
category: CUDA
draft: false
pin: 0
toc: true
lang: ""
abbrlink: cuda-indexing
---
本文从数据并行计算的角度出发，介绍 CUDA 中 Grid、Block 和 Thread 的分层组织方式，并解释如何通过线程索引将计算任务映射到具体数据。

## 关键概念

CUDA 的思想是：把一个数据并行的大任务拆成大量小任务，让 GPU 上的执行单元并行处理，类似 SIMD。一次 CUDA kernel launch 会产生一个 grid，grid 由多个 block 组成，block 由多个 thread 组成。
- **Grid**：一次 kernel launch 对应的整体执行任务
- **Block**：grid 的分块，是被调度到 SM 上执行的基本单位
- **Thread**：程序员视角下的最小执行单位，每个 thread 通常负责一小部分数据

在 kernel 内部，每个 thread 可以通过其坐标计算自己在整个 grid 中的全局坐标，从而确定自己负责处理哪一段数据。

下图展示了一个二维 grid，每个 block 内又包含二维 thread 结构：

![](Attachments/CUDAIndexing.png)

**CUDA Kernel** 是一个用 `__global__` 声明的函数。
- 它在 GPU (device) 上执行，由 CPU (host) 发起调用。
- 执行指定操作，例如执行 GEMM 操作或者 vector_add 等等

```cpp
// definition
__global__ void myKernel() {
	...
}

// invoke
dim3 gridDim(2, 2);
dim3 blockDim(4, 3);
mykernel<<<gridDim, blockDim>>>();
```

**Grid**. 当 Host 启动一个 kernel 时，CUDA runtime 会在 Device 上创建一个 **grid**。
- 一个 grid 是由多个 thread blocks 组成的集合。
- 每一次 kernel launch 会生成一个 grid，Kernel launch 与 grid 之间是一一对应的关系。
- 不同的 kernel launch 可以使用不同的 grid 维度和 block 维度。
- `gridDim.{x,y,z}` 表示 Grid 的 shape

**Block**. 多个 thread 组成的集合。
- 在同一个 grid 中，所有的 block 的 shape 都是一样的
- 在不同 kernel launch 对应的 grid 中，block 的 shape 可能不同
- `blockDim.{x,y,z}` 表示 Block 的 shape
- 使用 `blockIdx.{x,y,z}` 表示当前 block 在 grid 中的索引

**Thread**. 实际执行的最小 (software) execution unit.
- 使用 `threadIdx.{x,y,z}` 表示当前 thread 在 block 中的索引
- **每一个 block 最多只能有 1024 个 threads**. 线程可以按 1D/2D/3D 方式组织（如 (16,16)、(32,8)），但本质限制是线程总数，且通常选择 32 的倍数以匹配 warp 执行。[^1]

[^1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

Grid 和 Block 都可以是三维的，如下图所示：
![](Attachments/CUDAIndexing3D.png)

## Thread indexing

并行计算中，不同线程需要处理不同的数据。为了确定每个线程所负责的数据位置，我们需要计算每个线程的 **global thread index**。在这个章节我们会研究如何通过线程索引来确定每个线程处理的数据位置。

![定位 thread index 的例子](Attachments/FlattenedIndexingIllustration.png)

总体思路：目的是将多维的 thread 组织（grid/block/thread）映射为一维的 global thread index
- 首先定位 block 位置，把 blockIdx flatten 成 1D，确认每个 block 有多少个 thread
- 再确定 thread 在 block 第几行，把 threadIdx flatten 成 1D，确认 block 中的每一行有多少个 thread
- 最后加上 threadIdx.x

$$
\text{threadId} = \underbrace{\text{blockId}}_{\text{grid 内 block 的线性编号}} \times \underbrace{\text{threadsPerBlock}}_{\text{每个 block 的线程总数}} + \underbrace{\text{localThreadId}}_{\text{block 内 thread 的线性编号}}
$$

![](Attachments/1DGrid1DBlock.png)

1D Grid + 1D Block 情况：总共有 $\text{blockDim}.x \times \text{gridDim}.x$ 个 thread，
$$
\text{threadId} = (\text{blockIdx}.x \times \text{blockDim}.x) + \text{threadIdx}.x
$$

![](Attachments/1DGrid2DBlock.png)

1D Grid + 2D Block 情况：总共有 $\text{gridDim}.x \times (\text{blockDim}.x \times \text{blockDim}.y)$ 个 thread，
$$
\text{threadId} = (\text{blockIdx}.x \times \text{blockDim}.x \times \text{blockDim}.y) + (\text{threadIdx}.y \times \text{blockDim}.x) + \text{threadIdx}.x
$$

![](Attachments/2DGrid1DBlock.png)

2D Grid + 1D Block 情况：总共有 $(\text{gridDim}.x \times \text{gridDim}.y) \times \text{blockDim}.x$ 个 thread，
$$
\text{blockId} := (\text{blockIdx}.y \times \text{gridDim}.x) + \text{blockIdx}.x
$$
因此
$$
\text{threadId} = \text{blockId} \times \text{blockDim}.x + \text{threadIdx}.x
$$

![](Attachments/2DGrid2DBlock.png)

2D Grid + 2D Block 情况：总共有 $(\text{gridDim}.x \times \text{gridDim}.y) \times (\text{blockDim}.x \times \text{blockDim}.y)$ 个 thread,
$$
\text{blockId} = \text{gridDim}.x \times \text{blockIdx}.y + \text{blockIdx}.x
$$
因此
$$
\text{threadId} = \text{blockId} \times \text{blockDim}.x \times \text{blockDim}.y + \text{blockDim}.x \times \text{threadIdx}.y + \text{threadIdx}.x
$$

## 参考资料

- [CUDA Thread Indexing](https://anuradha-15.medium.com/cuda-thread-indexing-fb9910cba084)
- [Mastering CUDA Kernel Development: A Comprehensive Guide](https://medium.com/@omkarpast/mastering-cuda-kernel-development-a-comprehensive-guide-1f3032666b94)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
