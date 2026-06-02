---
title: 在无外网远程机器上使用 HuggingFace 或 Codex
published: 2026-05-01T12:04:46.686Z
description: |-
  在某些环境中远程机器无法直接访问外网，导致无法在远程机器上下载 HuggingFace 模型、无法调用 OpenAI / Codex API，pip install 或 git clone 失败等等情况。我们希望能够让远程机器借用自己本地电脑的网络来对这些网站和服务器进行访问。

  本文将介绍基于这个思路的解决方案和实现方式。本质上，我们通过 SSH 隧道把远程服务器的请求转运到本地，再由本地去访问互联网，最后把结果再返回给远程。
updated: 2026-05-30T12:29:29Z
tags:
  - Misc
category: Misc
draft: false
pin: 0
toc: true
lang: ""
abbrlink: reverse-proxy
---
在某些环境中远程机器无法直接访问外网，导致无法在远程机器上下载 HuggingFace 模型、无法调用 OpenAI / Codex API，pip install 或 git clone 失败等等情况。我们希望能够让远程机器借用自己本地电脑的网络来对这些网站和服务器进行访问。

本文将介绍基于这个思路的解决方案和实现方式。本质上，我们通过 SSH 隧道把远程服务器的请求转运到本地，再由本地去访问互联网，最后把结果再返回给远程。

## 思路

核心思路是**利用 SSH 隧道 + SOCKS 代理，让远程服务器的所有网络请求经由本地发出。**

![](Attachments/reverse-proxy-config.png)

数据流可以理解为：

```
remote server
   ↓ (proxy)
127.0.0.1:1080 (remote)
   ↓ (SSH -R 隧道)
127.0.0.1:1080 (local)
   ↓ (SOCKS5 代理)
Internet
```

假设我能够通过 `ssh remoteMachine` 来连接到我的远程服务器。

**第一步：在本地启动 SOCKS5 代理**。首先在我的本地电脑上配置：在本地启动一个 SOCKS5 代理服务器，监听 1080 端口。之后，任何发送到这个端口的请求，都会由本地机器代为访问互联网。
- 注意这一步执行完之后命令行不会有任何输出，但请不要关闭，保持其后台运行

```sh
uvx --from proxy-py proxy --hostname 127.0.0.1 --port 1080
```

**第二步：建立远程到本地的端口映射**。最重要的一步，将这个端口与远程服务器端口做端口映射，把远程服务器的 127.0.0.1:1080 映射到你本地的 127.0.0.1:1080。这样，当 remote 试图访问 127.0.0.1:1080 时，实际访问的是我本地机器的 127.0.0.1:1080.
- 注意这一步执行完之后命令行不会有任何输出，但请不要关闭，保持其后台运行

```bash
ssh -v -N -o ExitOnForwardFailure=yes \
	-R 127.0.0.1:1080:127.0.0.1:1080 remoteMachine
```

:::note
更推荐的做法是直接开启一个 SSH Config 配置项，例如：
```
Host i167-proxy
    HostName 172.16.0.167 # <-- replace
    Port 20000 # <--- replace
    User lindeyi # <--- replace
    ProxyJump slurm
    ForwardAgent yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
    RemoteForward 127.0.0.1:1080 127.0.0.1:1080 # <--- replace 
    ExitOnForwardFailure yes
    ForkAfterAuthentication yes
    SessionType none
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

然后通过 `ssh i167-proxy` 连接。
:::

**第三步：在远程配置代理环境变量**。让所有 HTTPS 请求，不要直接出去而是先交给 127.0.0.1:1080 这个代理。

```bash
// 写入 ~/.zshrc 或者 ~/.bashrc
export ALL_PROXY=socks5h://127.0.0.1:1080
export HTTPS_PROXY=socks5h://127.0.0.1:1080
export HTTP_PROXY=socks5h://127.0.0.1:1080
```

这样就完成了。

## Codex 身份验证

在完成上述代理配置后，远程服务器已经能够正常访问 HuggingFace 等网站，但此时仍然可能无法直接使用 Codex。原因是 Codex CLI 需要登录账号，而登录流程通常依赖浏览器；在某些远程服务器或受限网络环境中，登录过程可能会因为地区检测或浏览器不可用而失败。

解决方法是：先在本机完成 Codex 登录，确保本机可以正常使用 Codex。登录成功后，Codex 会在本机保存 credential / token / session 等认证信息。随后，我们可以将这些认证信息导出，并导入到远程服务器上。

请参考：[VS Code Remote-SSH 学校远程服务器使用 Codex 指南](https://zhuanlan.zhihu.com/p/2027512416271968050) 文章。