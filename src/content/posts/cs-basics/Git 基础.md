---
title: Git 基础
published: 2026-03-24T19:37:40.472Z
description: ""
updated: ""
tags:
  - CS-Basics
draft: false
pin: 0
toc: false
lang: ""
abbrlink: git-basics
---
## Git 底层数据模型

### 文件系统相关：Blob, Tree 和 Snapshot

现在假设我们的项目的文件树 (tree) 如下：

```
.
├── docs
│   ├── someDoc1.md
│   └── someDoc2.md
├── main.py
├── pyproject.toml
└── README.md
```

可以看到这棵文件树由子树和文件构成。

![](Attachments/TreeBlob.png)
在 Git 中，文件树中的文件和目录对应 Git 的两种对象：
- **Blob**：文件内容（纯数据）
- **Tree**：目录结构（名字 + 指针）



在本例中：

```
(root tree)
├── docs        → tree (T_docs)
├── main.py     → blob (B_main)
├── pyproject.toml → blob (B_pyproject)
└── README.md   → blob (B_readme)

// 其中
T_docs (tree)
├── someDoc1.md → blob (B_doc1)
└── someDoc2.md → blob (B_doc2)

// 再往下
B_main        → "print(...)"
B_pyproject   → "[tool.poetry] ..."
B_readme      → "# Project ..."
B_doc1        → "Some documentation..."
B_doc2        → "More documentation..."
```

**快照（Snapshot）**. 在项目不断演化的过程中，追踪的根目录下的文件和目录结构都有可能发生改变.我们可以将某一时刻被追踪的根目录（包括文件内容和目录结构）视为项目的一个完整状态，这个状态称为一个 snapshot。

### 历史追踪相关：History 和 Commit

**History**. 在 Git 中，历史可以表示为一个有向无环图（DAG），其中每个节点是项目的一个 snapshot.
- 这意味着，除了根节点之外（即初始状态），每个 snapshot 都会指向一个或多个父节点
- 这意味着这个 snapshot 由之前项目的每个状态经过若干改变修正而来


**Commit**. Commit 是 Git 历史中的一个节点，它包含一个 snapshot（即一个 root tree），以及指向其父 commit 的引用，从而将多个 snapshot 连接成一个有向无环图。

![](Attachments/GitCommit.png)

Git 中的 commit 是不可变的。这不意味着 commit 中的错误无法被修正，只是说，对提交历史的“修改”，实际上是创建新的 commit，然后更新引用（reference）去指向这些新的 commit。

## 总结

Commit, tree 和 blob 的概念可以从下图体现。Commit 通过存储项目根目录 tree 指针的方式记录项目的 snapshot.


![](Attachments/TreeBlob2.png)

## Git 中的引用

Commit 通过 SHA-1 哈希值存储，这不适合人类记忆。为了方便记忆，Git 创建了 **引用 (reference)** 的概念，其中重要的是：
- Branch 是指向某个 commit 的可变引用
- HEAD，是当前当前所在位置的引用
	- 通常情况下，你应该处在某个 branch 的某个 commit 上，而不是直接指向某个 commit，因此应该是 `HEAD -> branchName -> C`
	- 在特殊情况下，即直接指向某个 commit，称之为 detached HEAD：`HEAD -> C`

## Git Repository

基于上文介绍的概念，我们所称之为 **Git repository** 由两部分组成：
- **对象（objects）**：包括 blobs、trees 和 commits，用于表示文件内容、目录结构以及项目的快照
- **引用（references）**：包括 HEAD 和各个 branch，用于指向某个 commit，从而确定当前所处的位置

## Branching 艺术

## 紧急情况处理

## 参考资料

- [Version Control and Git](https://missing.csail.mit.edu/2026/version-control/)
- [Part 2: Blobs and trees](https://alexwlchan.net/a-plumbers-guide-to-git/2-blobs-and-trees/)