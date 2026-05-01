---
title: LLM 推理知识大纲
published: 2026-04-20T09:05:08.527Z
description: ""
updated: ""
tags:
  - LLM-Infra
draft: false
pin: 99
toc: true
lang: ""
abbrlink: llm-infra
---
## 分布式并行策略

- Data Parallellism (DP).
	- DDP
	- FSDP / Zero
- Tensor Parallelism (TP).
	- [TP 实现](大模型推理并行策略.md#TP%20实现)
- Pipeline Parallelism (PP).
- Sequence Parallelism (SP).
	- 和 TP 相结合
- Context / Sequence Parallel for Long Context
	- Context Parallelism (CP)
	- [DeepSpeed Ulysses：Attention Block 的序列并行](DeepSpeed%20Ulysses：Attention%20Block%20的序列并行.md)
	- [Ring Attention：Attention Block 的序列并行](Ring%20Attention：Attention%20Block%20的序列并行.md)
	- USP
- Expert Parallelism (EP)
	- EPLB
- 组合并行.

## Attention 优化

- FlashAttention v1/v2/v3/v4
	- [Online Softmax 推导](Online%20Softmax%20推导.md)
	- [Flash Attention (FA1)](Flash%20Attention%20(FA1).md)
- Flash Decoding
- PagedAttention
- Ring Attention
- online softmax / merge 思想
- Sparse / Sliding Window / Linear Attention（可选）

## KV Cache 管理

- PagedAttention
- [Prefix Cache：前缀 KV Cache 缓存](Prefix%20Cache：前缀%20KV%20Cache%20缓存.md)
- Cache eviction / reuse
- Hybrid KV Cache Manager
- KV transfer

## 调度优化

- Static batching
- [Continuous Batching](Continuous%20Batching.md)
- [Chunked Prefill](Chunked%20Prefill.md)
- Prefill-Decode 协同调度
- 长短请求混部
- Admission control / priority

## Serving 架构
- Offline vs Online inference
- 单实例 / 多实例
- Router / Scheduler / Worker
- PD 分离
- Disaggregated Prefill
- Multi-model / Multi-LoRA serving

## MoE 系统优化
- Router
- Top-k gating
- Expert Parallelism
- all-to-all cost
- grouped GEMM / fused MoE kernels
- load balance / expert placement / replication

## 模型压缩与部署成本优化

- Quantization
- KV cache quantization
- Distillation
- LoRA / Multi-LoRA serving

## 通信与系统基础

- [并行计算基础通信原语](../parallelism/并行计算基础通信原语.md)
- 通信-计算 overlap
- 带宽模型与性能分析
