---
title: "Halo: Batch Query Processing and Optimization for Agentic Workflows (2601)"
published: 2026-06-10T14:20:12.644Z
description: ""
updated: ""
tags:
  - Agent-Infra
  - 论文阅读笔记
category: Agent-Infra
draft: true
pin: 0
toc: true
lang: ""
abbrlink: 2601-halo
---
## **目标**

Halo 的目标是把数据库里的“batch query processing / query optimization”思想迁移到 agentic workflows 上。它不是只优化单次 LLM 调用，而是把一批 agent workflow 视为可优化的 DAG 查询计划，在全局层面合并重复计算、复用上下文和 KV cache、协调 CPU 工具调用与 GPU LLM 推理，从而提升批量 agent 任务的端到端 latency 和 online serving throughput。 

## **问题**

文章认为，现有 LLM serving engine 主要优化单个 request，例如 continuous batching、attention kernel、speculative decoding 等；但它们不知道整个 agent workflow 的结构，也不知道不同 workflow 之间有哪些共享 prompt、共享上下文、共享 SQL/API/tool call。因此在 batch analytics 场景下，大量 agent 会重复执行相似的 LLM prompt、相似的 SQL 查询、相似的 retrieval/API 调用，造成严重冗余。 

另一方面，LangGraph、AutoGen、AgentScope 这类 multi-agent framework 能表达 workflow，但主要负责 orchestration，不掌握底层 serving runtime 的 GPU 状态、KV cache、模型加载、CPU-GPU pipeline 等系统信息。所以它们能“把任务跑起来”，但不能做 plan-level optimization。结果是 LLM 节点和 Tool 节点被割裂调度，CPU 工具调用可能阻塞 GPU，GPU 也可能因为等待工具结果而空转。 

更本质的问题是：agentic workflow 不是单一请求，而是带依赖结构的异构 DAG。节点既有 GPU 上的 LLM inference，也有 CPU/I/O 上的 SQL、HTTP、parsing、本地函数调用。传统 request-level serving 只看“请求队列”，看不到 DAG 结构；传统 DAG 系统如 Spark/Ray 又把 LLM 当黑盒，看不到 KV cache、prefill/decode、模型权重 residency 这些 LLM-specific 状态。 

## **Key insights / Observation**

第一，batch agent workflows 之间存在大量结构性冗余。多个 agent 可能执行相同 SOP、相同子图、相似 prompt、相似 SQL 查询或相同 retrieval call。只要把它们合并成一个全局图，就能发现跨 workflow 的共享计算，而不是每个 agent 各跑各的。 

第二，agent workflow 的优化单位应该从 request 升级为 DAG plan。也就是说，系统不应该只问“下一个 LLM request 怎么 batch”，而应该问“这一批 workflow 的 LLM 节点、Tool 节点、依赖关系、模型放置、KV cache 复用，整体怎么排”。这是 Halo 相比普通 LLM serving 的核心视角转变。 

第三，CPU-GPU heterogeneity 是关键瓶颈。Agent workflow 经常在 CPU-bound tool 和 GPU-bound LLM 之间切换。如果 CPU 工具调用没有提前准备好，GPU 会出现 pipeline bubble；如果 GPU 调度不考虑模型权重和 KV cache 状态，就会产生 model switching、prefill 重算和 cache miss。 

第四，LLM operator 是 stateful 的。一个 LLM 节点的执行成本不只由 prompt 长度决定，还取决于 worker 上当前是否已经加载对应模型、是否有可复用的 KV cache、是否能共享 prefix。换句话说，同一个节点放到不同 GPU worker 上，成本可能完全不同。 

## **解决的方法**

Halo 把每个 agent workflow 表示成 structured query-plan DAG，其中 LLM 节点通常运行在 GPU 上，Tool 节点通常运行在 CPU 上。然后它把一批 workflow 合并成 consolidated graph，用这个全局图暴露跨 workflow 的共享结构、重复 prompt、重叠上下文和重复工具调用。 

系统由 Parser、Optimizer、Processor 三部分组成。Parser 把 YAML workflow 转成 typed GraphSpec，并把嵌在 prompt 或 agent logic 里的 SQL/API/local function 抽取成独立 Tool 节点。这样 CPU 工具调用不再是 LLM prompt 里的黑盒副作用，而是可以被调度和优化的 operator。 

Optimizer 做全局 plan-level scheduling。它用 cost model 同时估计 CPU preparation cost、model loading cost、LLM prefill/decode cost、KV cache reuse benefit，以及 worker placement 的影响。然后用 epoch-based dynamic programming 在 DAG frontier 上搜索，把 LLM 节点分配到 GPU worker，并决定执行顺序，目标是减少 makespan、系统总负载和碎片化调度开销。 

Processor 负责把优化计划真正执行出来。它维护 GPU workers 和 CPU workers，做 adaptive batching、KV-cache sharing/prefetching、tool request coalescing、CPU-GPU overlap。对于相同 SQL/API/tool call，它会合并成一次物理执行，再把结果 fan-out 给多个逻辑节点；对于 LLM 节点，它尽量保持模型和上下文 locality，让同模型、共享 prefix、共享 KV 的请求在合适的 worker 上连续执行。