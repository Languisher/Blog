---
title: LLM 基础
published: 2026-03-01T15:24:05.272Z
description: ""
updated: ""
tags:
  - LLM
draft: true
pin: 0
toc: true
lang: zh
abbrlink: llm-basic
---
本章将会

## 引入

### Encoding: Neural Modeling

假设 $w_{1:n}$ 是一个 sequence，其中每一个 $w_{i}\in \mathcal{V}$ 都是有限集字典的一个元素。你可以简单理解为每个 $w_{i}$ 代表一个英文单词或者是一个 token id.

我们将字典中的每一个元素映射为在 $\mathbb{R}^d$ 中的高维向量。由于需要对字典的每一个元素都建立映射规则，因此定义对应的 embedding matrix $$
E \in \mathbb{R}^{d \times |\mathcal{V}|}
$$
因此对于我们的 sequence，其 embedding 是 $x = E w$，且：
$$
x_{1:n} = w_{1:n} E^\top \in \mathbb{R}^{n \times d}
$$

这里我们得到的是对于每个 $w_{i}$，其对应的 non-contextual embedding $x_{i} \in \mathbb{R}^d$. 在此基础上，我们需要结合该 token 的上下文信息，得到其对应的 contextual embedding $h_{i} \in \mathbb{R}^d$，如果是自回归模型的话则与该 token 之前的 embedded tokens 相关。


### Sentence Encoding: mean-pooling as an example

在 RNN 最为流行的时代，为了通过一段文字得到二分类评价（比如评价一个评论是正面还是负面的）往往采取 Sentence Encoding：通过将一句话中的词向量做平均 (mean-pooling)，接一个分类器，最终得到正/反二分结果。这种做法每个词向量的权重都是一样的，不考虑词序也不考虑语义重要性。
![](Attachments/mean-pooling-rnn.png)

## Attention: weighted sum of values

基于上文提到的 mean-pooling 思想，Attention 做了进一步优化：Attention 对于上下文中参与计算的 embedding 使用不同的权重。

### 权重计算：Lookup Table 的思想



### 加权求和

每个上下文 token 现在有两个向量需要表示：
- $K$ 值：与 Query 进行点乘计算，以得到该 token 的权重值
- $V$ 值：实际表示该 token 意义，即原先的 embedding 的意义

