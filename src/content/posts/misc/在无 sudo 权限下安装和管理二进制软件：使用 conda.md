---
title: 在无 sudo 权限下安装和管理二进制软件：使用 conda
published: 2026-04-26T12:10:46.250Z
description: ""
updated: ""
tags:
  - Misc
category: Misc.
draft: false
pin: 0
toc: false
lang: ""
abbrlink: conda-manage-binary-software
---
我们希望在公司或学校的服务器上安装自己需要的软件包。然而，大多数服务器都基于 Debian 系 Linux 发行版，而系统默认通常只提供 `apt` 作为包管理工具。和其他系统级包管理器一样，`apt` 需要 **sudo 权限** 才能安装软件。

但现实中，让系统管理员为某个用户安装一些冷门工具，往往既不方便，也不太合理。一个典型的例子是 `zsh`：虽然它是一个相当常见的 shell，但在很多服务器上默认并没有安装。在这种情况下，似乎唯一的选择就是从源码手动编译。

不过，其实还有一个更优雅的办法：使用 conda。虽然 conda 通常被认为是 Python 的环境和包管理工具，但它同样可以管理通用的二进制软件。特别是 **conda-forge** 这个社区维护的 channel，提供了大量已经编译好的软件包。

```sh
conda install conda-forge::<package_name>
# 或
conda install -c conda-forge package_name

# 示例
conda install -c conda-forge zsh
```

## 参考资料

[Miniconda：无需 sudo 权限的软件包管理工具](https://zhuanlan.zhihu.com/p/688624885)
