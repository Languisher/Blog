---
title: LLM 基础：Attention
published: 2026-03-01T15:24:05.272Z
description: ""
updated: ""
tags:
  - LLM
draft: false
pin: 0
toc: true
lang: zh
abbrlink: llm-basic-1
---
本章将会

> 参考论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 引入

### Encoding: Neural Modeling

假设 $w_{1:n}$ 是一个 sequence，其中每一个 $w_{i}\in \mathcal{V}$ 是词表中的一个 token，可以将其理解为一个 token id 或者一个英文单词。为了进行矩阵运算，我们通常将 $w_i$ 表示为对应的 one-hot 向量：

$$
w_i \in \{0,1\}^{|\mathcal{V}|}.
$$

我们将字典中的每一个元素映射为在 $\mathbb{R}^d$ 中的高维向量。由于需要对字典的每一个元素都建立映射规则，因此定义对应的 embedding matrix 

$$
E \in \mathbb{R}^{d \times |\mathcal{V}|}
$$

因此对于我们的 sequence，其 embedding 是 $x = E w$，且：
$$
x_{1:n} = w_{1:n} E^\top \in \mathbb{R}^{n \times d}
$$

这里我们得到的是对于每个 $w_{i}$，其对应的 non-contextual embedding $x_{i} \in \mathbb{R}^d$. 在此基础上，我们通常通过一个序列建模结构（例如 RNN 或 Transformer），结合上下文信息计算每个 token 的 contextual embedding $h_i \in \mathbb{R}^d$.

在自回归模型中，$h_i$ 仅依赖于该 token 之前的 tokens。


### Sentence Encoding：平均池化

在 RNN 最为流行的时代，为了通过一段文字得到二分类评价（比如评价一个评论是正面还是负面的）往往采取 Sentence Encoding：例如，可以通过 mean-pooling 得到句子表示：
$$
h_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} h_i
$$

这种做法每个词向量的权重都是一样的，不考虑词序也不考虑语义重要性。，接一个分类器，最终得到正/反二分结果。这种做法每个词向量的权重都是一样的，不考虑词序也不考虑语义重要性。
![](Attachments/mean-pooling-rnn.png)

## Attention：对所有“词向量”的加权平均

基于上述 mean-pooling 的思想，Attention 可以看作一种 **数据依赖的加权 pooling**：不同的上下文 token 会被分配不同的重要性权重，而不是均匀平均。
### 权重计算：Lookup Table 的思想

Attention 本质上就是一种**加权平均**——当这些权重是通过学习得到时，这种机制会变得非常强大！

如下图，在 **查找表（lookup table）** 中，
- 我们有一组 key 映射到对应的 value。给定一个 query，它会精确匹配某一个 key，从而返回对应的 value
- 这是 one-hot 选择
- 用公式可以表示为：

$$
O = \sum_i \alpha_i V_i, \quad \alpha_i \in \{0,1\}, \quad \sum_i \alpha_i = 1
$$

![](Attachments/lookup-table.png)

而在 **Attention** 中：
- **query** 会对所有 **key** 进行“软匹配”，得到一组介于 0 和 1 之间的权重
- 每个 key 对应的 **value** 会乘以对应权重，然后进行加权求和


![](Attachments/attention-illu.png)

每个上下文 token 现在有两个向量需要表示：
- $K$ 值：与 Query 进行点乘计算，以得到该 token 的权重值
- $V$ 值：实际表示该 token 意义，即原先的 embedding 的意义

### Attention 公式表达

考虑序列 $x_{1:n}$ 中的一个 token $x_i$。我们定义其对应的查询向量（query）为：
$$
q_i = Q x_i,\quad Q \in \mathbb{R}^{d \times d}
$$

对于序列中的每一个 token $x_j \in \{x_1, \dots, x_n\}$，我们分别定义对应的键（key）和值（value）为：
$$
k_j = K x_j,\quad v_j = V x_j
$$
其中 $K \in \mathbb{R}^{d \times d}$，$V \in \mathbb{R}^{d \times d}$。

那么，对于序列中**任意 token $x_i$ 的上下文表示（contextual representation）**  $h_i$，也就是 attention 的输出，是对整个序列中 value 的加权线性组合：
$$
\boxed{h_i = \sum_{j=1}^{n} \textcolor{red}{\alpha_{ij}} v_j}
$$

其中权重 $\alpha_{ij}$ 表示第 $j$ 个 token 对 $x_i$ 的贡献强度（重要性）。

这些权重的计算方式如下：
- 首先计算 query $q_i$ 与所有 key $\{k_1, \dots, k_n\}$ 的相似度（affinity），通常使用点积 $q_i^T k_j$
- 然后在整个序列维度上做 Softmax 归一化

因此，对于序列中**任意两个 token $x_i$ 和 $x_j$，其注意力权重 $\alpha_{ij}$** 表示在计算 $x_i$ 的上下文表示时，第 $j$ 个 token 对其贡献的强度，其计算方式为：
$$
\boxed{\alpha_{ij} = \frac{\exp(q_i^T k_j)}{\sum_{t=1}^{n} \exp(q_i^T k_t)}}
$$