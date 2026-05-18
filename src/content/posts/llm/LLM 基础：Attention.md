---
title: LLM 基础：Attention
published: 2026-03-01T15:24:05.272Z
description: 本文将会介绍 LLM 的核心部件 Attention.
updated: 2026-05-06T07:07:35Z
tags:
  - LLM
  - Attention
draft: false
pin: 0
toc: true
lang: zh
abbrlink: llm-basic-1
---
本文将会介绍 LLM 的核心部件 Attention.

> 参考论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 引入

假设输入是一段 token 序列：

$$
w_{1:n} = (w_1, w_2, \dots, w_n).
$$

在现代语言模型中，原始文本先经过 tokenizer，被切分并映射为一串 token ids：

$$
\text{text} \xrightarrow{\text{tokenizer}} (t_1, t_2, \dots, t_n),
$$

其中每个 $t_i \in \{0,1,\dots,|\mathcal V|-1\}$ 表示词表 $\mathcal V$ 中的一个 token id。

接下来，模型会通过一个 embedding table 将每个 token id 映射到一个 $d$ 维向量空间中。记 embedding table 为：

$$
E =
\begin{bmatrix}
- E_1 - \\
- E_2 - \\
\vdots \\
- E_{|\mathcal V|} -
\end{bmatrix}
\in \mathbb{R}^{|\mathcal V| \times d},
$$

那么第 $i$ 个 token 的 embedding 可以直接理解为从 embedding table 中取出第 $t_i$ 行：$x_i = E[t_i] \in \mathbb{R}^d.$

对于整个序列，我们可以得到：

$$
X =
\begin{bmatrix}
x_1 = E[t_{1}] \\
x_2 = E[t_{2}] \\
\vdots \\
x_n = E[t_{n}]
\end{bmatrix}
\in \mathbb{R}^{n \times d}.
$$

这里得到的是每个 token 的 **non-contextual representation**：

$$
\boxed{x_i \in \mathbb{R}^d.}
$$

所谓 non-contextual，是指此时 $x_i$ 只由 token id 本身决定，不依赖它出现在什么句子、什么上下文中。也就是说，只要 token id 相同，从 embedding table 中查到的初始向量就是相同的。

但语言理解真正困难的地方在于：一个 token 的含义往往不是孤立决定的，而是由它所在的上下文共同决定的。因此，在得到初始 embedding 之后，我们还需要通过一个序列建模结构，例如 RNN 或 Transformer，让不同位置的 token 之间发生信息交互。

经过上下文建模之后，第 $i$ 个位置会得到新的表示：
$$
\boxed{h_i \in \mathbb{R}^d.}
$$

这里的 $h_i$ 就是 **contextual representation**。它不再只由当前 token 本身决定，而是由当前 token 以及它能够看到的上下文共同决定。

在自回归语言模型中，由于预测当前位置时不能看到未来信息，因此 $h_i$ 通常只能依赖当前位置及其之前的 tokens：

$$
h_i = f(x_1, x_2, \dots, x_i).
$$

这里我们不展开介绍 RNN 是如何实现的，通常情况下每个 token 的 contextual representations 是根据以下方式得到的：

$$

h_i = \mathrm{RNN}(x_i, h_{i-1}),

$$

### RNN Sentence Encoding：平均池化

在 RNN 最为流行的时代，为了通过一段文字得到二分类评价（比如评价一个评论是正面还是负面的）往往采取 Sentence Encoding：例如，可以通过 mean-pooling 得到句子表示：
$$
h_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} h_i
$$

这种方法的核心问题在于：它本质上是在对所有 token 的表示做平均。这意味着每个 token 都被赋予相同权重，模型无法区分哪些词更重要，也无法动态关注与当前任务最相关的信息。与此同时，mean pooling 本身也会压缩序列结构，使得词序信息和长距离依赖容易被弱化。
![](Attachments/mean-pooling-rnn.png)

## Attention：对所有“词向量”的加权平均

基于上述 mean-pooling 的思想，Attention 可以看作一种 **数据依赖的加权池化**：不同的上下文 token 会被分配不同的重要性权重，而不是均匀平均。
### 权重计算：Lookup Table 的思想

Attention 本质上可以理解为一种**加权平均（weighted aggregation）**。而当这些权重能够根据输入内容动态学习时，这种机制就会变得非常强大。

为了理解 Attention，可以先从传统的 **查找表（lookup table）** 出发。无论是编程语言中的字典（dictionary），还是更底层的哈希表，本质上都在维护一种 key->value 的映射关系。

如下图所示：
- 我们拥有一组 key，每个 key 对应一个 value
- 当给定一个 query 时，系统会找到与之匹配的 key
- 随后返回该 key 对应的 value

传统 lookup table 的特点在于：query 最终只会匹配到一个 key，因此本质上是一种 **one-hot selection**。用 $\alpha$ 表示这个过程，用公式可以表示为：

$$
O = \sum_{i \in \text{Lookup Table Key}} \alpha_i V_i, \quad \alpha_i \in \{0,1\}, \quad \sum_i \alpha_i = 1
$$

![](Attachments/lookup-table.png)

而在 **Attention** 中：
- **query** 会对所有 **key** 进行“匹配”，得到一组介于 0 和 1 之间的权重。 这个权重值由 query 和 key 的值决定。这一步意味着 $\alpha$ 不再是 one-hot selection.
- 每个 key 对应的 **value** 会乘以对应权重，然后进行加权求和

![](Attachments/attention-illu.png)

每个上下文 token 现在有两个向量需要表示：
- $K$ 值：与 Query 进行点乘计算，以得到该 token 的权重值。$K$ 决定了该不该关注这个 token
- $V$ 值：表示该 token 实际提供给上下文聚合的信息内容

### Attention 公式表达

考虑序列 $x_{1:n}$ 中的一个 token $x_i \in \mathbb{R}^{d \times 1}$。我们定义其对应的查询向量（query）为：
$$
q_i = Q x_i,\quad Q \in \mathbb{R}^{d \times d}
$$

> 注：这里为了方便数学推导，将单个 token 表示为列向量形式。在实际深度学习框架中，每个 tensor activation 的形状是 $[b,s,d]$，因此每个 token 是行向量。

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

为了让模型能够在所有上下文 token 之间动态分配注意力，以及让输出尺度稳定，我们通常希望这些权重形成一个概率分布，这些权重的计算方式如下：
- 首先计算 query $q_i$ 与所有 key $\{k_1, \dots, k_n\}$ 的相似度（affinity），通常使用点积 $q_i^T k_j$
- 然后在 key 维度上做 Softmax 归一化

因此，对于序列中**任意两个 token $x_i$ 和 $x_j$，其注意力权重 $\alpha_{ij}$** 表示在计算 $x_i$ 的上下文表示时，第 $j$ 个 token 对其贡献的强度，其计算方式为：
$$
\boxed{\alpha_{ij} = [\text{softmax}(q_{i}K^T)]_{j} = \frac{\exp(q_i^T k_j)}{\sum_{t=1}^{n} \exp(q_i^T k_t)}}
$$

这里：
- 分子表示当前 query 与第 $j$ 个 token 的匹配程度
- 分母则是在所有 token 上进行归一化，使得所有注意力权重之和为 1

进一步地，对于单个 query $q_i$，我们可以得到其对应的完整注意力分布：

$$
\boxed{
\alpha_i
=
(\alpha_{i1}, \alpha_{i2}, \dots, \alpha_{in})
}
$$

它描述了当前 token 在理解自身时，会从整个上下文中的哪些 token“读取”多少信息。

### 可解释性（Interpretability）

注意力机制提供了一定程度的可解释性：
- 通过观察注意力分布，可以看到模型在关注哪些位置
- 可以“免费”获得（soft）对齐关系（alignment）
- 模型会自行学习这种对齐关系

注意力分布示例：

![图中的每一行表示一个 query 对整个输入序列的注意力分布，横轴表示 key/value 所对应的输入 token，颜色越深，表示注意力权重越高。这里展示的是一个机器翻译任务。可以看到，当模型生成法语中的 “il” 时，其注意力主要集中在英文中的 “he” 上](Attachments/attention_alignment.png)

## Self-Attention

在普通的 Attention 计算公式中，$Q$ 与 $(K, V)$ 的来源没有任何约束，它们可以来自不同的表示。
$$
\text{Attention}: (Q, K, V) \mapsto \text{softmax}(Q K^T)V
$$

**Self-Attention（自注意力）** 是一种特殊情况，其中 $Q, K, V$ 都来自同一个输入序列。给定输入序列表示 $X$：
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$
## 序列顺序问题 - 位置编码（Position Embedding）

自注意力本身不包含顺序信息，因此我们需要**显式地编码序列中的位置信息**，使 Attention 不仅能判断哪些 token 时相关的，还有基于它们的相对位置判断（例如“它、他、她”指代谁？）并将其融入到 $Q, K$ 中。

> 对 $W_V$ 并不严格必要，因为 $W_V x_i$ 本身是非上下文的。

将每个位置表示为一个向量：
$$
\forall i \in \{1, ..., n\} : p_i \in \mathbb{R}^d
$$

设 $x_i$ 是 token $w_i$ 的 embedding，则带位置的表示通常为：
$$
\text{Embed}(x, i) = x_i + p_i \in \mathbb{R}^d
$$

> 也可以拼接（concatenate），但实践中通常采用相加（add）。

### 正弦位置编码（Sinusoidal Position Encoding）

使用不同频率的正弦和余弦函数来编码位置：

![sin_pe](Attachments/sin_pe.png)

$$
p_i =
\begin{bmatrix}
\sin(i / 10000^{2j/d}) \\
\cos(i / 10000^{2j/d}) \\
\vdots \\
\sin(i / 10000^{2(d/2)/d}) \\
\cos(i / 10000^{2(d/2)/d})
\end{bmatrix}
$$

**优点**
- 周期性意味着模型更关注相对位置，而不是绝对位置
- 理论上可以外推到更长序列

**缺点**
- 不可学习
- 实际外推效果往往不好

### 可学习位置编码

核心思想：
$$
\forall i \in \{1, ..., n\}, \quad p_i
$$
作为可学习参数，构成矩阵：
$$
p \in \mathbb{R}^{d \times n}
$$

每个 $p_i$ 是一列。

**优点**
- 灵活性高，可以适配数据

**缺点**
- 无法外推到训练范围之外的位置

### 现代位置编码 - RoPE

设 embedding 函数为：
$$
f(x, i)
$$

#### 核心思想

希望表示满足：
- 对**绝对位置不敏感**
- 只依赖**相对位置**

即：
$$
\langle f(x, i), f(y, j) \rangle = g(x, y, i - j)
$$

但传统的：
$$
f(x, i) = x_i + p_i
$$
会引入 $\langle p_i, y \rangle$, $\langle x, p_j \rangle$, $\langle p_i, p_j \rangle$ 等包含了绝对位置信息的变量。

#### 基于旋转的方案

定义旋转操作：
$$
f(x, i) = R_i x_i
$$

其中 $R_i$ 是正交变换，满足：
$$
R_i^T = R_{-i}
$$

则有：
$$
\langle R_i x, R_j y \rangle = \langle x, R_{j-i} y \rangle
$$
只依赖相对位置，可以看到旋转操作很好满足了我们希望编码函数只包含相对位置关系的要求。

#### 示例：二维旋转

$$
R_i =
\begin{bmatrix}
\cos \theta_i & -\sin \theta_i \\
\sin \theta_i & \cos \theta_i
\end{bmatrix}, \quad
\theta_i = i \omega
$$

问题在于：
$$
R_i = R_j \iff (i - j)\omega = 2\pi k
$$

因此单个频率的旋转编码具有周期性，**在长序列下可能发生位置冲突（collision）。** 例如取 $\omega = \frac{\pi}{1024}$，则位置 $j=0$ 和 $i=2048$ 对应相同旋转，模型无法仅通过该二维旋转区分他们的位置。虽然可以通过减小 $\omega$ 来增大周期，从而缓解长距离冲突问题，但更小的 $\omega$ 会导致 $R_i \approx R_{i+1}$，即相邻位置之间的旋转差异变小，从而降低局部位置分辨率。

#### RoPE 的解决方法

因此，RoPE 并不是只使用一个旋转频率，而是在不同二维子空间中使用不同频率进行旋转，也就是说，RoPE 会将一个高维 embedding 拆成多个二维子空间，并分别执行：
$$
	\text{RoPE}(x,i) = \text{diag}(R_{i}^{(\omega_{j})})_{j \in \left[ 1, \dots, \frac{d}{2} \right]} x_{i}
$$

![](Attachments/rope_rotation.png)

将位置编码为高维“相位向量”，这意味着
- 高频子空间对局部位置变化更敏感
- 低频子空间具有更长的周期，可以表示更远距离关系

### 数学形式

对 Q 和 K 使用同样的 RoPE 旋转方式：
$$
\begin{align}
f_{Q}(x_m, m) &= R_{\Theta, m}^d W_{Q} x_m \\
f_{K}(x_m, m) &= R_{\Theta, m}^d W_{K} x_m
\end{align}
$$

其中：
$$
R_{\Theta, m}^d =
\begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & \cdots & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & \cdots & 0 \\
0 & 0 & \cos(m\theta_2) & \cdots & 0 \\
0 & 0 & \sin(m\theta_2) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \cos(m\theta_{d/2}) \\
0 & 0 & 0 & \cdots & \sin(m\theta_{d/2})
\end{bmatrix}
$$

$$
\theta_k = 10000^{-2k/d}
$$
![](Attachments/rope_highdim.png)

## 非线性层避免多个 Attention 退化

如果堆叠两层 attention：
$$
\begin{aligned}
o_i &= \sum_{j=1}^n \alpha_{ij} V^{(2)} \left( \sum_{k=1}^n \alpha_{jk} V^{(1)} x_k \right) \\
&= \sum_{k=1}^n \left( \alpha_{jk} \sum_{j=1}^n \alpha_{ij} \right) (V^{(2)} V^{(1)}) x_k
\end{aligned}
$$
> 这里 $\alpha$ 分别指代不同层的权重结果，不同层的权重结果是不一致的，这里只是简写。

如果两层 attention 之间没有非线性的话，多层会退化成一层.

因此我们需要在每一层 attention 后，对每个 token 独立应用 MLP：
$$
\begin{aligned}
m_i &= \text{MLP}(\text{output}_i) \\
&= W_2(\text{ReLU}(W_1 \cdot \text{output}_i + b_1) + b_2)
\end{aligned}
$$

![](Attachments/attention_nonlinear.png)

## 未来信息屏蔽 Masking

在自回归建模中：
$$
w_t \sim \text{softmax}(f(w_{1:t-1}))
$$

预测当前位置时，不能看到未来信息。

通过 mask 实现：
$$
\alpha_{ij}^{\text{masked}} =
\begin{cases}
\alpha_{ij}, & j \le i \\
-\infty, & \text{otherwise}
\end{cases}
$$
![](Attachments/attention_futuremasking.png)

现代大模型推理系统通常不显式构造 mask 矩阵，而是直接读取 Query 向量分别需要和哪些 Key 和 Value 向量进行计算。
