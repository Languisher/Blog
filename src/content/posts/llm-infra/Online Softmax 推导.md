---
title: Online Softmax 推导
published: 2026-04-20T15:55:30.843Z
description: Online softmax 通过维护可增量更新的中间状态 $(m,l)$，将原本需要全局两次归约（max 与 sum）的计算转化为可流式处理的形式，从而避免额外遍历，并使 softmax 支持分块计算与归并。
updated: ""
tags:
  - LLM-Infra
  - Algorithms
draft: false
pin: 0
toc: true
lang: ""
abbrlink: online-softmax
---
> 参考论文：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

Online softmax 通过维护可增量更新的中间状态 $(m,l)$，将原本需要全局两次归约（max 与 sum）的计算转化为可流式处理的形式，从而避免额外遍历，并使 softmax 支持分块计算与归并。

## Safe Softmax

**Safe softmax 函数**定义如下. 对于一个 vector $x \in \mathbb{R}^d$，为了避免指数爆炸，定义：
$$
m(x) := \max_{i}x_{i}, \quad \text{softmax}(x) := \frac{\begin{bmatrix}
\dots  & e^{x_{i}-m(x) } & \dots
\end{bmatrix}}{\sum_{j=1}^d e^{x_{j}-m(x)}} \in \mathbb{R}^d
$$

为了方便表示，对于任意局部子向量 $x_{1:k} := [x_1,\dots,x_k] \in \mathbb{R}^k.$，我们定义：
$$
\text{softmax}(x_{1:k}) = \frac{f_{k}(x)}{l_{k}(x)}
$$
其中：
$$
\begin{cases} 
m_{k}(x) &= \max_{i \in \{1, \dots, k\}} x_{i} \in \mathbb{R}\\
f_{k}(x) &= \begin{bmatrix}
e^{x_{1}- m_{k}(x)}  &  \dots & e^{x_{k}-m_{k}(x)}
\end{bmatrix} \in \mathbb{R}^k \\
l_k(x) &=  \sum_{j=1}^k e^{x_{j}-m_{k}(x)} \in \mathbb{R}
\end{cases}
$$

分子分母都与所有元素的最大值 $m_{k}(x)$ 有关。

在最朴素的实现中，softmax 通常需要 **三次遍历向量** $x$：
- 第一次遍历 $x$，以计算出 $x$ 的最大值 $m_{d}(x)$
- 第二次遍历 $x$，以得到  softmax 运算中的归一化因子 $l_{d}(x)$
- 第三次遍历 $x$，逐元素得到其分子部分

若只统计对输入向量 $x$ 的读取次数，则约为 $3d$ 次读取；若进一步将输出写回也计入内存访问，则总计约为 $4d$ 次内存访问。
## Online Softmax 算法推导：三次遍历优化成两次遍历

下图展示了 online softmax 的核心算法计算方式，在本章剩下的部分将会推导 Algorithm 3.


![](Attachments/OnlineSoftmaxAlgo.png)


Softmax 可以采用增量的方式来减少一次遍历，在一次遍历中就同时完成 $m_{d}(x)$ 和 $l_{d}(x)$ 的计算。

假设我们希望从前面 $j-1$ 个元素 $[x_{1},\dots, x_{j-1}]$ 的 *softmax 计算中间状态* 推导 $j$ 个元素 $[x_{1},\dots,x_{j}]$ 的 softmax 结果.
我们定义：
$$
\text{softmax}([x_{1},\dots, x_{j-1}]) = \frac{f_{j-1}(x)}{l_{j-1}(x)}
$$

我们现在希望推导：
$$
\text{softmax}([x_{1},\dots, x_{j}]) = \frac{f_{j}(x)}{l_{j}(x)}
$$

- **更新所有元素的最大值**：$m_{j}(x)$
- **更新归一化因子** $l_{j}(x)$：
	- $l_{j-1}(x)$ 部分由于所有元素最大值发生变化，因此需要进行 rescale
	- 再加上 $x_{j}$ 对应的计算值


因此 

$$
\begin{align}
m_{j}(x) &:= \max{(m_{j-1}(x), x_{j})}  \\
l_{j}(x) &:= l_{j-1}(x) \times \boxed{e^{m_{j-1}(x) - m_{j}(x)}} + e^{x_{j }-m_{j}(x)}
\end{align}
$$


这启发我们能够只需要两次遍历 $x\in \mathbb{R}^d$ 就可以实现 softmax 计算：
- 第一遍遍历：按照上面推导的公式迭代更新得到 $m_{d}(x)$ 和 $l_{d}(x)$
- 第二遍遍历：按照下面的计算方法得到实际的 softmax 结果
$$
\text{softmax}(x) := \frac{\begin{bmatrix}
\dots  & e^{x_{i}\boxed{-m_{d}(x) }} & \dots
\end{bmatrix}}{\boxed{l_{d}(x)}} \in \mathbb{R}^d
$$


这样 softmax 的计算只需要 **两次遍历向量** $x$，若若只算输入读取总的内存访问量约为 $2d$。

## 自顶而上思路：拆分向量为多个子向量，分别计算中间状态 $(m,l)$ 并最终合并

我们希望计算
$$
\operatorname{softmax}(x), \qquad x \in \mathbb{R}^{d}, \quad d = nk,\quad (n,k)\in\mathbb{N}^2.
$$

将向量 $x$ 按顺序拆分为 $n$ 个长度为 $k$ 的子向量：
$$
x = \bigl[x^{(1)},x^{(2)},\dots,x^{(n)}\bigr], \qquad x^{(i)}\in\mathbb{R}^{k}.
$$

Online softmax 先在每个子向量上分别计算局部中间状态，再将这些局部状态逐步合并，最终恢复出整个向量的 softmax 结果。

### 每个子向量的局部中间状态

现在考虑拆分后的第 $i$ 个子向量 $x^{(i)}\in\mathbb{R}^k$。  
我们先独立计算它的局部中间状态：
$$
m^{(i)} := \max_{1\le t\le k} x_t^{(i)},
$$
$$
l^{(i)} := \sum_{t=1}^{k} e^{x_t^{(i)} - m^{(i)}}.
$$

这里：

- $m^{(i)}$ 是第 $i$ 个子向量内部的最大值；
- $l^{(i)}$ 是该子向量在以 $m^{(i)}$ 为基准时的局部指数和。

于是有
$$
\sum_{t=1}^{k} e^{x_t^{(i)}} = e^{m^{(i)}} l^{(i)}.
$$

这说明，状态 $(m^{(i)}, l^{(i)})$ 足以概括子向量 $x^{(i)}$ 对全局 softmax 分母的贡献。

### 两个子向量状态的合并公式

接下来考虑两个向量块 $a$ 和 $b$，它们各自已经计算出了中间状态：
$$
(m_a, l_a), \qquad (m_b, l_b).
$$

我们希望得到拼接向量 $[a,b]$ 对应的中间状态 $(m,l)$。

**最大值**显然是两者最大值中的较大者：
$$
m = \max(m_a, m_b).
$$

**归一化因子**：根据定义，
$$
\sum_{x\in[a,b]} e^x
=
\sum_{x\in a} e^x + \sum_{x\in b} e^x.
$$

而对于每个子块，有
$$
\sum_{x\in a} e^x = e^{m_a} l_a, \qquad
\sum_{x\in b} e^x = e^{m_b} l_b.
$$

因此
$$
\sum_{x\in[a,b]} e^x = e^{m_a} l_a + e^{m_b} l_b.
$$

另一方面，如果我们希望用合并后的最大值 $m$ 来表示整体的中间状态，那么应当有
$$
\sum_{x\in[a,b]} e^x = e^m l.
$$

两边相等，于是得到
$$
e^m l = e^{m_a} l_a + e^{m_b} l_b.
$$

两边同时除以 $e^m$，可得
$$
l = l_a e^{m_a-m} + l_b e^{m_b-m}.
$$

所以，两个中间状态的合并规则为
$$
\boxed{
m = \max(m_a,m_b), \qquad
l = l_a e^{m_a-m} + l_b e^{m_b-m}.
}
$$


由于上述合并规则对任意两个块都成立，因此我们可以将其递归地应用到 $n$ 个子向量上，最终得到整个向量的全局状态 $(m,l)$

### 恢复 softmax


当我们得到整个向量的全局状态 $(m,l)$ 之后，softmax 的每个元素都可以写为
$$
\boxed{\operatorname{softmax}(x)_j
=
\frac{e^{x_j-m}}{l}.}
$$

## 总结

Softmax 的计算并不一定非要逐元素进行。  如果我们把输入向量拆分成多个子向量，再把每个子向量压缩为一个中间状态 $(m,l)$，那么整个 softmax 的分母就可以通过这些局部状态逐步合并得到。

因此，softmax 具有如下结构特征：
1. **块内可独立计算**：每个子块 $i$ 可以单独计算局部 $(m^{(i)},l^{(i)})$；
2. **块间可递归合并**：多个子块的状态可以通过统一公式归约：$m = \max(m_a,m_b), \;l = l_a e^{m_a-m} + l_b e^{m_b-m}.$
3. **最终可恢复输出**：全局状态 $(m,l)$ 足以恢复所有 softmax 分量：$\operatorname{softmax}(x)_j=\frac{e^{x_j-m}}{l}.$

与此同时这个状态合并运算满足结合律，因此不仅可以按顺序在线更新，也可以在**多个块之间并行归约**。这正是 online softmax 能支持分块计算、树形规约和并行实现的根本原因。