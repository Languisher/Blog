---
title: 在无外网远程服务器上使用 HuggingFace / Codex
published: 2026-05-01T12:04:46.686Z
description: |-
  在很多实际环境中远程机器往往无法直接访问外网。这会带来一系列问题，例如无法下载 HuggingFace 模型、无法调用 OpenAI / Codex API，甚至连 pip install 或 git clone 都可能失败。我们希望能够让服务器“借用”我本地的网络来对这些网站和服务器进行访问。

  本文将介绍基于这个思路的解决方案和实现方式。本质上，我们通过 SSH 隧道把远程服务器的请求转运到本地，再由本地去访问互联网，最后把结果再返回给远程。
updated: ""
tags:
  - Misc
draft: false
pin: 0
toc: true
lang: ""
abbrlink: reverse-proxy
---
在很多实际环境中远程机器往往无法直接访问外网。这会带来一系列问题，例如无法下载 HuggingFace 模型、无法调用 OpenAI / Codex API，甚至连 pip install 或 git clone 都可能失败。我们希望能够让服务器“借用”我本地的网络来对这些网站和服务器进行访问。

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
ssh -N -D 127.0.0.1:1080 localhost
# localhost can be replaced as some other hosts...
```

**第二步：建立远程到本地的端口映射**。最重要的一步，将这个端口与远程服务器端口做端口映射，把远程服务器的 127.0.0.1:1080 映射到你本地的 127.0.0.1:1080。这样，当 remote 试图访问 127.0.0.1:1080 时，实际访问的是我本地机器的 127.0.0.1:1080.
- 注意这一步执行完之后命令行不会有任何输出，但请不要关闭，保持其后台运行

```
ssh -N -R 127.0.0.1:1080:127.0.0.1:1080 remoteMachine
```

**第三步：在远程配置代理环境变量**。让所有 HTTPS 请求，不要直接出去而是先交给 127.0.0.1:1080 这个代理。

```
// 写入 ~/.zshrc 或者 ~/.bashrc
export ALL_PROXY=socks5h://127.0.0.1:1080  
export HTTPS_PROXY=socks5h://127.0.0.1:1080  
export HTTP_PROXY=socks5h://127.0.0.1:1080
```

这样就完成了。

## Codex 身份验证

在完成上述代理配置后，远程服务器已经能够正常访问 HuggingFace 等网站，但此时仍然可能无法直接使用 Codex。原因是 Codex CLI 需要登录账号，而登录流程通常依赖浏览器；在某些远程服务器或受限网络环境中，登录过程可能会因为地区检测或浏览器不可用而失败。

一种可行的解决方法是：先在本机完成 Codex 登录，确保本机可以正常使用 Codex。登录成功后，Codex 会在本机保存 credential / token / session 等认证信息。随后，我们可以将这些认证信息导出，并导入到远程服务器上。

可以使用这个脚本完成认证信息迁移：
https://github.com/chuvadenovembro/script-to-use-codex-cli-on-remote-server-without-visual-environment

  
假设可以通过 `ssh remoteMachine` 连接远程服务器，操作流程如下：

```sh
# 本地机器
# 先 git clone 仓库
# 如果遇到权限问题，先 chmod a+x ./codex-auth-transfer.sh
./codex-auth-transfer.sh export -o codex-auth.tar.gz

# 传到服务器
scp codex-auth.tar.gz remoteMachine:~

# 远程服务器
# 同样先 git clone 仓库
# 如果遇到权限问题，先 chmod a+x ./codex-auth-transfer.sh
# 如果压缩包和 git clone 仓库所在的位置不同，记得修改 -f 之后内容为实际相对/实际路径
./codex-auth-transfer.sh import -f codex-auth.tar.gz --force
```