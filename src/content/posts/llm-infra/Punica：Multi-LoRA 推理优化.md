---
title: Punica：Multi-LoRA 推理优化
published: 2026-05-04T09:48:36.969Z
description: Punica 将 Multi-LoRA 推理中按请求执行的 for-loop 重写为分段聚合的矩阵向量计算（SGMV），在一个融合 kernel 中同时处理多个 LoRA Adapter，从而避免小规模 GEMV 的频繁调度，显著提升 GPU 利用率。
updated: ""
tags:
  - LLM-Infra
  - LLM-LoRA
category: LLM 推理优化
draft: false
pin: 0
toc: true
lang: ""
abbrlink: punica
---
Punica 将 Multi-LoRA 推理中按请求执行的 for-loop 重写为分段聚合的矩阵向量计算（SGMV），在一个融合 kernel 中同时处理多个 LoRA Adapter，从而避免小规模 GEMV 的频繁调度，显著提升 GPU 利用率。

> 参考论文：[Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)

## Multi-LoRA 问题

在 [LoRA 和 Multi-LoRA：模型参数微调](../llm/LoRA%20和%20Multi-LoRA：模型参数微调.md) 中，我们将 Multi-LoRA 推理形式化为：
$$
h = Wx, \quad W = W_0 + \sum_i \alpha_i \Delta W_i
$$
然而在实际推理中，这一表达需要细化到 **token 级别**：对于 batch 中的每个输入 $x_j$，其对应的 LoRA Adapter（即 $\{\alpha_i^{(j)}, \Delta W_i\}_i$）可能不同。因此，不同 token 实际使用的权重扰动是不同的，这使得原本可以统一表示的矩阵乘法退化为一组“每个样本使用不同权重”的不规则计算。

这意味着我们需要一个 for-loop 来执行，这里假设每个 token 只激活一个 LoRA：
```
h_result = W_0 @ x

for token_j in x:
    lora_id = routing(token_j)   # 每个 token 选择一个 LoRA adapter
    h_lora = B[lora_id] @ (A[lora_id] @ token_j)
    h_result[j] += h_lora
```

For-loop 会导致大量小规模矩阵向量乘法与 kernel 调度开销，严重影响 GPU 利用率。

## Punica

Punica 将这种逐 token / 逐 adapter 的不规则计算重新组织为一个 fused segmented GEMV（SGMV）操作。

从逻辑上看，可以将 batch 中的 token 按其对应的 LoRA adapter 进行分组，即对于每个 adapter $i$，收集其对应的输入集合 $\{x_j\}$。然而，Punica 并不会对每个 adapter 显式执行一个 for-loop 并分别调用 GEMM。

相反，Punica 通过一个融合的 SGMV kernel，在单次 kernel 调用中并行处理所有 token。对于每个 token $x_j$，根据其对应的 adapter id $\ell_j$，动态选择对应的参数 $(A_{\ell_j}, B_{\ell_j})$，并完成如下计算：

$$
h_j = B_{\ell_j}(A_{\ell_j} x_j)
$$
- 和统一的 GEMV 的区别：不同 token 对应的参数不同，需要在 runtime 动态推导对应参数
- 在每次 kernel 调用中，并不会对 $A_l, B_l$ 进行显式的重排或组合，而是根据每个 token 对应的 LoRA adapter id，在计算过程中动态索引并访问对应的参数。

该过程在 GPU 内部以 token 为粒度并行执行，从而避免了逐 adapter 的多次 kernel 调度，实现了对不规则计算的高效融合。
