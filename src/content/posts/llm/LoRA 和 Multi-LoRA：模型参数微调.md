---
title: LoRA 和 Multi-LoRA：模型参数微调
published: 2026-05-04T09:06:50.273Z
description: LoRA 在原模型权重上 $W$ 引入一个低秩结构的增量项（$\Delta W = BA$），在不改变原参数的情况下，仅训练该低秩参数，从而在低维子空间中实现高效微调。
updated: ""
tags:
  - LLM
draft: false
pin: 0
toc: true
lang: ""
abbrlink: lora
---
> TBD：更多 LoRA 类型。

**LoRA** 在原模型权重上 $W$ 引入一个低秩结构的增量项（$\Delta W = BA$），在不改变原参数的情况下，仅训练该低秩参数，从而在受限的低维子空间中实现高效微调。

**Multi-LoRA** 在一个共享的底座模型上同时引入多个 LoRA Adapter，并通过组合多个低秩扰动使模型能够在多个领域之间进行灵活切换或融合，提升其多任务适配能力。

> 参考论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## 微调

微调任务可以建模为在原始权重 $W_{0}\in \mathbb{R}^{d \times k}$ 上学习一个增量。假设原来前向是 $h = W_{0}x$，加上增量$\Delta W\in \mathbb{R}^{d \times k}$ 之后：
$$
	h = (W_{0}+ \Delta W)x
$$
若直接学习完整的 $\Delta W \in \mathbb{R}^{d \times k}$，则等价于进行全参数微调，其参数量与原模型相同，带来较高的计算与存储开销，我们希望减少训练的参数量。

## SVD 低秩分解

> TBD：如果有时间，会写一篇 SVD 低秩分解的推导过程文章

对于任意矩阵 $\Delta W \in \mathbb{R}^{d \times k}$，根据 SVD 分解，有：

$$
\Delta W = U \Sigma V^{T}
$$

如果 $\Delta W$ 本身近似低秩，那么它可以用一个秩为 $r$ 的矩阵近似表示：

$$
\Delta W \approx U_r \Sigma_r V_r^{T}
$$

其中 $r \ll \min(d, k)$。

## LoRA

LoRA 的核心思想是：在微调时，不直接训练一个完整的扰动矩阵 $\Delta W \in \mathbb{R}^{d \times k}$，而是假设模型参数的更新量具有低秩结构，将其参数化为两个小矩阵的乘积：

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)
$$




因此，原来的前向传播变为：

$$
h = (W_0 + \Delta W)x = W_0x + BAx
$$

在训练时，原始权重 $W_0$ 被冻结，只训练 $A$ 和 $B$。由于 $A$ 和 $B$ 的参数量为 $r(d+k)$，而完整扰动矩阵 $\Delta W$ 的参数量为 $dk$。

当 $r \ll \min(d,k)$ 时，LoRA 需要训练的参数量远小于直接微调整个矩阵。

## Multi-LoRA：多个 LoRA 组合

LoRA 可以在固定底座模型的基础上，通过引入低秩扰动 $\Delta W = BA$，实现对特定任务或场景的高效微调。然而，在多任务场景下（例如代码、医疗、对话等），如果每个任务对应一个独立的 LoRA Adapter，则推理时通常需要在不同 Adapter 之间切换，这既限制了模型的多任务能力，也带来额外的系统开销。

**Multi-LoRA** 在一个共享的底座模型上同时引入多个 LoRA Adapter，并通过组合多个低秩扰动使模型能够在多个领域之间进行灵活切换或融合，提升其多任务适配能力。具体而言：
- 每个任务对应一个 LoRA Adapter，即一组低秩参数 $(A_i, B_i)$，对应扰动 $\Delta W_i = B_i A_i$
- 推理时，通过权重 $\alpha_i$ 对多个 LoRA Adapter 进行加权组合：

$$  
W = W_0 + \sum_i \alpha_i \Delta W_i  
$$
- 权重 $\alpha_i$ 可以是固定的、由用户指定的，或根据输入动态计算

这种方式使模型能够在多个任务子空间之间进行连续组合，而不仅仅是离散切换，从而提升多任务适配能力。

## 参考资料

- [从LoRA到Multi-LoRA：原理&代码实践](https://zhuanlan.zhihu.com/p/1984729458444363168)