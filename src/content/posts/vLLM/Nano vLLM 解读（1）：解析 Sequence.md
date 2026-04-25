---
title: Nano vLLM 解读（1）：解析 Sequence
published: 2026-04-19T13:24:46.446Z
description: 本文介绍了在 Nano-vLLM 系统中基本请求执行单位 `Sequence` 的实现。
updated: ""
tags:
  - LLM-Infra
  - Nano-vLLM
draft: false
pin: 0
toc: true
lang: ""
abbrlink: nano-vllm-sequence
---
本文介绍了在 Nano-vLLM 系统中基本请求执行单位 `Sequence` 的实现。
## 请求的基本单位：Sequence 类

> 本小节对应 Nano vLLM 的 [sequence.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/sequence.py) 和 [`class Sequence`](https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/sequence.py#L14)，点击链接跳转。

在 Nano-vLLM 中，单个请求由 `Sequence` 类封装，在 LLMEngine 的各个部件之间传递。Sequence = 数据（tokens） + 状态（status）+ 调度信息（scheduler metadata）+ 推理策略（sampling params）


![](Attachments/Sequence.png)


`Sequence` 类实例的创建 API：

```python
# https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/llm_engine.py#L43-L46
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
	if isinstance(prompt, str):
		prompt = self.tokenizer.encode(prompt)
		
	# prompt: list[int], Tokenized token id 列表
	# sampling_params: 采样策略
	seq = Sequence(prompt, sampling_params)
	
	# ...
```

在构造 Sequence 的过程中：
- 为每个请求分配唯一 seq_id（用于调度与跟踪）
- 初始化推理状态（如 WAITING）以支持后续状态流转
- 规范化并缓存输入 token 信息（token_ids、长度、last_token）
- 初始化调度与 KV cache 相关元数据（如 block_table、cached_tokens）
- 绑定采样参数（temperature / max_tokens 等），使其成为完整的推理执行单元

```python
# https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/sequence.py#L14
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0    # tokens that don't need prefill
        self.num_scheduled_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

## Sequence 类与 Engine 其它部件之间的联系

- 如何被调度，什么时候执行 Prefill/Decode：由 Scheduler 决定，详见 [Nano vLLM 解读（3）：解析 Scheduler](Nano%20vLLM%20解读（3）：解析%20Scheduler.md)
- 其逻辑层面的 KV Cache Block 
	- 逻辑分配/释放：由 Scheduler 的子模块 BlockManager 决定，详见[Nano vLLM 解读（4）：解析 BlockManager](Nano%20vLLM%20解读（4）：解析%20BlockManager.md)
	- 如何与实际机器上的 KV Cache Block 联系起来：由 ModelRunner 决定，详见