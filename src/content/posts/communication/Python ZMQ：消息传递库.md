---
title: Python ZMQ：消息传递库
published: 2026-05-19T11:30:48.882Z
description: ""
updated: ""
tags:
  - Python
  - 通信
category: 通信
draft: false
pin: 0
toc: true
lang: ""
abbrlink: python-zmq
---
## 不同客户端之间通信

在 ZMQ 中，每个客户端（进程）都需要先加入 ZMQ 的消息系统，然后才能进行通信。对于每个客户端来说，通常需要完成三件事：
1. 创建自己的 ZMQ runtime，可以将其理解为一个轻量级的消息通信操作系统[^1]。
2. 创建并注册消息端点（socket），声明自己的通信角色
3. 接入同一个通信网络（bind/connect）

[^1]: 它位于应用层与底层 TCP socket 之间，负责管理消息队列、socket、后台 I/O 线程以及网络连接状态，并处理异步消息收发、自动重连、消息路由等通信逻辑，从而将底层复杂的网络通信机制封装起来，对上层应用暴露更加简单统一的 `send/recv` 消息接口以及更高层的通信语义（如 REQ/REP、PUB/SUB 等通信模式）。

整个结构可以表示为：
```
Process
 └── Context
      └── Socket
           └── bind/connect
```

其中：
- `Context` 表示当前进程中的 ZMQ runtime
- `Socket` 表示一个消息通信端点
- `bind/connect` 表示将通信端点接入网络

```python
import sys
import zmq


class Server:
    def __init__(
        self,
        address: str = "tcp://localhost:5555"
    ):

        # 【1】创建当前进程中的 ZMQ runtime
        self.context = zmq.Context.instance()

        # 【2】创建一个 Reply 类型的消息端点
        self.socket = self.context.socket(zmq.REP)

        # 【3】绑定到网络地址，对外提供服务
        self.socket.bind(address)

    def run(self):
        while True:
            try:
                # 接收客户端消息
                message = self.socket.recv()

                print(f"recv: {message}")

                # 回复客户端
                self.socket.send(b"ok")

            except KeyboardInterrupt:
                sys.exit(0)
```