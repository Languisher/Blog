---
title: LLM Infra 知识大纲
published: 2026-04-20T09:05:08.527Z
description: ""
updated: ""
tags:
  - LLM-Infra
draft: true
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
	- Pipeline Parallelism (PP).
- Sequence Parallelism (SP).
	- 和 TP 相结合
- Context / Sequence Parallel for Long Context
	- Context Parallelism (CP)
	- DeepSpeed Ulysses
	- Ring Attention
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
- Prefix Caching
- Cache eviction / reuse
- Hybrid KV Cache Manager
- KV transfer

## 调度优化

- Static / Dynamic batching
- Continuous Batching
- Chunked Prefill
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
- NCCL 与集合通信
- all-reduce / all-gather / reduce-scatter / all-to-all
- 通信-计算 overlap
- 拓扑感知
- 带宽模型与性能分析

## 框架与系统
### 训练框架
- Megatron / Megatron-Core
- DeepSpeed
- NeMo
- FSDP / ZeRO 生态

### 推理框架
- vLLM
- TensorRT-LLM
- SGLang
- TGI / LMDeploy（按需）

