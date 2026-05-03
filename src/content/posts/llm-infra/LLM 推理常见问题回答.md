---
title: LLM 推理常见问题回答
published: 2026-05-01T18:31:18.489Z
description: ""
updated: ""
tags:
  - LLM-Infra
draft: true
pin: 0
toc: true
lang: ""
abbrlink: llm-infra-interview
---
## Inference

**在大语言模型推理中，prefill 和 decode 有什么区别，以及为什么这对系统设计很重要？**

Prefill 阶段会在一次并行的前向传播中处理整个输入 prompt，因此它是计算受限的（compute-bound），因为我们需要将较大的 token 矩阵与权重矩阵进行乘法运算。

Decode 阶段则以自回归的方式逐个生成 token，因此是内存带宽受限的（memory-bandwidth-bound），因为在每一步生成单个 token 时，我们都需要加载全部模型权重。

这种区别对系统设计至关重要：prefill 和 decode 在最优 batch size、GPU 利用率特征以及对延迟的敏感性方面都有很大不同。

现代系统（例如 Splitwise 和 Mooncake）会将 prefill 和 decode 物理地分离到不同的 GPU 资源池中，使它们可以分别独立扩展和优化。


**什么是推理中的“铁三角”？在优化一个对延迟敏感的聊天应用时，你会如何做决策？**

推理中的“铁三角”描述了延迟（latency）、吞吐（throughput）和成本（cost）之间的关系——提升其中一个，通常会以牺牲另一个为代价。

对于一个对延迟敏感的聊天应用，我会优先关注 TTFT（首 token 时间）和 TPOT（每个输出 token 的时间）。具体来说：

1. 使用更小的模型或量化模型，以减少每一步的计算量
2. 保持较小的 batch size，或者使用带有严格 SLO 感知调度的 continuous batching
3. 使用 speculative decoding，在每一步中生成多个 token
4. 使用 prefix caching，对于重复的 system prompt 跳过 KV cache 的重复计算

作为权衡，我会接受更高的单 token 成本，以换取更低的延迟。

**为什么在大语言模型（LLM）的解码阶段，瓶颈通常是内存带宽，而不是原始计算能力（FLOP/s）？**

在小 batch size 的自回归解码过程中，GPU 在生成每一个 token 时，都必须从 HBM 中加载整个模型的权重。一个 7B 参数、FP16 精度的模型大小约为 14 GB；即使在带宽为 2 TB/s 的 A100 上，每一步也大约需要 7 ms。与此同时，相比 GPU 张量核心能够提供的计算能力，单个 token 的实际计算量微不足道。矩阵-向量乘法（解码阶段的关键操作）的算术强度大约只有 1–2 FLOP/byte，远低于达到计算受限所需的约 300 FLOP/byte 阈值。只有通过增大 batch size，才能摊销权重加载的成本，从而逐渐接近计算受限的行为。

**你如何衡量并报告推理性能？在一个生产级服务系统中，你会监控哪些指标？**
 
我会跟踪以下指标：
 - TTFT（首 token 时间，对用户感知的响应速度至关重要）
 - TPOT（每个输出 token 的时间，决定流式输出的平滑程度）
 - 吞吐量（所有用户的 tokens/second）
 - GPU 利用率和显存利用率
 - 有效吞吐（满足 SLO 的请求比例）
 - 队列深度（用于检测系统是否过载）
 - 以及每百万 token 的成本
 
 我会建立分位数（p50、p95、p99）监控面板，而不是只看平均值，因为真正影响用户体验的是尾延迟。此外，我还会分别跟踪 prefill 延迟和 decode 延迟，以便能够独立诊断各自的瓶颈。