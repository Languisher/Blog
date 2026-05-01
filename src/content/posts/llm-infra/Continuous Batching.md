---
title: Continuous Batching
published: 2026-05-01T17:29:43.639Z
description: Continuous batching 是一种面向在线推理的调度机制，在每个 decode step 动态组织当前活跃请求，从而避免 padding 和长尾带来的资源浪费，提高系统吞吐、提高 GPU 利用率.
updated: ""
tags:
  - LLM-Infra
draft: false
pin: 0
toc: true
lang: ""
abbrlink: continuous-batching
---

> [!note]
> **Continuous batching** 是一种面向在线推理的调度机制，在每个 decode step 动态组织当前活跃请求，从而避免 padding 和长尾带来的资源浪费，提高系统吞吐、提高 GPU 利用率.

直接将多个请求简单拼成一个固定 batch 进行推理，会带来两个结构性问题：
1. **长度不一致导致的 padding 浪费**：不同请求的输入长度不同。为了将它们组织成规则 tensor 输入模型，通常需要对较短序列进行 padding。模型在计算时也会对这些“无效 token”执行 attention 和 FFN 计算，造成算力浪费。
2. **Decode 阶段的生命周期不同步（Tail Latency）**：在生成阶段，每个请求需要生成的 token 数量不同，而且生成长度在开始时是不可预测的。在固定 batch 机制下，只有当 batch 中**最后一个请求完成生成**后，整个 batch 才能结束并返回结果。因此，已经提前完成生成的请求会被迫等待最长的那个请求完成，导致尾延迟（tail latency）显著上升。

![](Attachments/NaiveBatching.png)

核心思想是当一个请求结束（例如生成 `<EoS>` Token）立刻就被替换为新的请求：
![](Attachments/ContinuousBatchingGoal.png)

Continuous Batching 通常依赖于 **ragged batching** 技术来实现，即在同一个批次中拼接来自不同请求、长度不一的 token 序列。为了保证不同请求之间互不干扰，系统会通过构造块 block-diagonal 的 Causal Attention Mask（或等价的序列偏移管理机制），确保每个请求的 token 仅与自身历史 token 进行 Attention 计算，而不会与其他请求的 token 发生交互。

![](Attachments/RaggedBatching.png)

Continuous Batching 通常与 Chunked Prefill 结合使用。下图展示了在三个请求同时存在时的推理示例。在每一个调度 step 中，系统的执行流程如下：
- 对正在进行 Decode 的请求执行一次 forward 计算，各生成一个新的 token；
- 在剩余的计算预算内，调度处于 Prefill 阶段的请求，按 chunk 大小处理相应数量的 token；
- 当某个请求完成（Decode 结束）时，将其从当前 batch 中移除；
- 将新到达的请求或尚未完成 Chunked Prefill 的请求加入 batch，以维持持续的计算负载。

![](Attachments/ContinuousBatching.png)
Continuous Batching 在工程实现上通常结合 Paged Attention，以实现对 KV Cache 的更细粒度管理。若采用“一个 token 对应一个 block”的设计，会导致显存碎片化严重，并带来较高的元数据与调度开销。更高效的做法是使用固定大小的 Cache Block，每个 block 存储多个连续 token 的 KV 表示，从而在保持灵活分配能力的同时减少碎片化并提高内存利用率。
