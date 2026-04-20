---
title: Attention 模块优化 3：Flash Decoding (FA-V3)
published: 2026-04-19T14:49:36.954Z
description: ""
updated: ""
tags:
  - LLM-Infra
draft: false
pin: 0
toc: true
lang: ""
abbrlink: flash-decoding
---
在 Decode 阶段，由于 KV cache 很大，attention 计算受到 memory bandwidth 限制。  
Flash Decoding 通过将 KV 按块划分，并利用 **可合并的 softmax 计算方式**，使得不同块可以并行计算并在最后进行规约，从而实现高效的并行 attention 计算。