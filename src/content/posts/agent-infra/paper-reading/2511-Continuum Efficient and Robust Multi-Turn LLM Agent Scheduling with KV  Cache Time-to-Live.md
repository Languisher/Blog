---
title: "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV  Cache Time-to-Live"
published: 2026-06-10T14:39:52.283Z
description: ""
updated: ""
tags:
  - 论文阅读笔记
  - Agent-Infra
category: Agent-Infra
draft: true
pin: 0
toc: true
lang: ""
abbrlink: 2511-continuum
---
## **目标**

Continuum 的目标是：让多轮 Agent workload 在 LLM serving 中跑得更快、更稳。核心指标是降低 agent program 的整体完成时间，而不是只优化单次 LLM request 的 TTFT/TPOT。文章认为，Agent 不是一次请求，而是“LLM 推理 → 工具调用 → 新上下文 → 再次 LLM 推理”的连续程序，因此 serving system 应该保护这种多轮连续性。 

## **问题**

传统 vLLM/SGLang 这类 inference engine 默认采用 end-of-turn eviction：一次 decode 结束后，如果有新请求排队，就把当前 request 的 KV cache 释放掉。这对普通聊天没问题，因为人类下一轮输入通常很慢；但对 Agent 不合适，因为很多 tool call 很短，可能几毫秒到几秒后就回来继续推理。KV cache 一旦被释放，下一轮要么重新 prefill，要么从 CPU/DRAM reload，都会带来额外开销。 

更关键的是，文章指出 prior work 忽略了 per-turn queueing delay。即使 KV cache 已经 offload 到 CPU，回来时 reload 很快，请求也仍然要重新进入等待队列，等 GPU memory 空出来才能继续执行。这个等待会在每一轮工具调用后重复发生，多轮 Agent 会把这个 delay 累积放大。 

另一个问题是 tool call duration 高度不稳定。一直 pin KV cache 可以减少 miss，但如果工具调用突然很慢，KV cache 长时间占用 GPU memory，会阻塞其他请求，甚至造成 memory pressure 或 deadlock。因此简单的“永远保留”也不行。 

## **Key insights / Observation**

第一，Agent workload 的调度单位不应该只是单个 request，而应该是 program / trajectory。一次 Agent 任务由多个 LLM turn 和 tool call 组成，单次请求结束不代表整个任务结束。传统 serving 把每个 turn 当作独立请求，会破坏 Agent 的执行连续性。 

第二，KV cache 的价值在 Agent 场景中不会在 decode 结束后立刻归零。因为工具调用后，下一轮 LLM request 很可能复用前面的上下文。也就是说，tool call 期间的 KV cache 更像是“暂时休眠的状态”，而不是“已经完成的垃圾”。 

第三，KV cache retention 的收益不只是避免 prefill/reload，还包括避免重新排队。文章认为这是 InferCept 等方法的主要缺口：它们主要看 reload cost，但没有建模多轮 Agent 反复返回队列造成的累计等待。 

第四，保留 KV cache 需要有上限。工具调用时间是长尾分布，某些工具很快，但少数调用非常慢。如果没有 TTL，GPU memory 可能被长时间占住；如果 TTL 太短，又会刚好在 tool call 返回前释放，前面的等待白白浪费。所以关键不是“保留 or 不保留”，而是“保留多久”。 

## **解决的方法**

Continuum 的方法是给 Agent turn 结束后的 KV cache 引入 Time-to-Live，也就是 TTL。对于产生 tool call 的 LLM request，系统不会立刻释放它的 KV cache，而是根据收益和成本计算一个保留时间。在 TTL 内，如果下一轮 request 回来，就可以直接复用 GPU 中的 KV cache，并且优先继续调度该 program；如果 TTL 过期还没回来，就自动释放 KV cache，避免长期占用 GPU memory。 

它的 TTL 决策模型主要比较两件事：保留 KV cache 的收益，以及占用 GPU memory 的机会成本。收益包括 Prefill/Reload cost 和 Out-of-order / queueing cost；成本是这段时间内 KV cache 占用 GPU memory，导致其他请求无法使用这些资源。然后 Continuum 根据历史 tool-call duration 的经验分布，估计某个 TTL 内工具返回的概率，选择期望净收益最大的 TTL。 

在调度上，Continuum 还加入 TTL-aware priority：被 pin 且仍在 TTL 窗口内的 program 下一轮请求会被优先调度，从而维持 program-level continuity。同时它保留 program-level FCFS，使较早到达的 Agent program 不会因为工具调用而不断被后来请求插队。 

实现上，Continuum 在 vLLM 上增加了一个 Tool Call Handler，用来识别 tool call、记录工具执行时间、估计 TTL，并把 TTL hint 交给 scheduler。scheduler 负责 pin/unpin KV blocks，并在 TTL 过期或 program 结束时释放 KV cache。整体改动比较模块化，不需要重写整个 inference engine。