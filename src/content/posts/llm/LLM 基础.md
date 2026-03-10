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

假设 $w_{1:n}$ 是一个 sequence，其中每一个 $w_{i}\in \mathcal{V}$ 是词表中的一个 token，可以将其理解为一个 token id 或者一个英文单词。为了进行矩阵运算，我们通常将 $w_i$ 表示为对应的 one-hot 向量：

$$
w_i \in \{0,1\}^{|\mathcal{V}|}.
$$

我们将字典中的每一个元素映射为在 $\mathbb{R}^d$ 中的高维向量。由于需要对字典的每一个元素都建立映射规则，因此定义对应的 embedding matrix $$
E \in \mathbb{R}^{d \times |\mathcal{V}|}
$$
因此对于我们的 sequence，其 embedding 是 $x = E w$，且：
$$
x_{1:n} = w_{1:n} E^\top \in \mathbb{R}^{n \times d}
$$

这里我们得到的是对于每个 $w_{i}$，其对应的 non-contextual embedding $x_{i} \in \mathbb{R}^d$. 在此基础上，我们通常通过一个序列建模结构（例如 RNN 或 Transformer），结合上下文信息计算每个 token 的 contextual embedding $h_i \in \mathbb{R}^d$.

在自回归模型中，$h_i$ 仅依赖于该 token 之前的 tokens。


### Sentence Encoding: mean-pooling as an example

在 RNN 最为流行的时代，为了通过一段文字得到二分类评价（比如评价一个评论是正面还是负面的）往往采取 Sentence Encoding：例如，可以通过 mean-pooling 得到句子表示：
$$
h_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} h_i
$$

这种做法每个词向量的权重都是一样的，不考虑词序也不考虑语义重要性。，接一个分类器，最终得到正/反二分结果。这种做法每个词向量的权重都是一样的，不考虑词序也不考虑语义重要性。
![](Attachments/mean-pooling-rnn.png)

## Attention: weighted sum of values

基于上述 mean-pooling 的思想，Attention 可以看作一种 **数据依赖的加权 pooling**：不同的上下文 token 会被分配不同的重要性权重，而不是均匀平均。
### 权重计算：Lookup Table 的思想



### 加权求和

每个上下文 token 现在有两个向量需要表示：
- $K$ 值：与 Query 进行点乘计算，以得到该 token 的权重值
- $V$ 值：实际表示该 token 意义，即原先的 embedding 的意义

