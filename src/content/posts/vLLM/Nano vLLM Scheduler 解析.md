---
title: Nano vLLM Scheduler 解析
published: 2026-04-20T21:47:40.646Z
description: ""
updated: ""
tags:
  - vLLM
  - LLM-Infra
draft: true
pin: 0
toc: true
lang: ""
abbrlink: nano-vllm-scheduler
---
本文从 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 入手，解读一个轻量级的大模型请求 Scheduler 是如何做的。

## 请求的基本单位：Sequence 类

> 本小节对应 Nano vLLM 的 [`class Sequence`](https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/sequence.py#L14)，点击链接跳转。


![](Attachments/Sequence.png)

## LLM Engine 上层调度 API

> 本小节对应 Nano vLLM 的 [`Engine.generate()` API](https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/llm_engine.py#L60)，点击链接跳转。

在具体看 Scheduler 的实现细节之前，我们先研究一下 LLM Engine 是如何调用 Scheduler 的 API 的。用一张图来概览 LLM Engine 的调用流程：

![](Attachments/LLMEngine.png)

下面代码忽略了实现细节，只是为了表述整体流程。

`LLMEngine.generate()` 一次性接受多个请求，并在一次 `generate()` 调用内部，通过反复调用 `step()` 推进所有请求直到完成（这里不是流式输入，而是一次性提交一批请求）：
1. 将输入请求
	1. Tokenize
	2. 封装成 Sequence 对象
	3. 调用 `scheduler.add()` 加入 Scheduler waiting queue
2. 当 Scheduler 仍有请求没有完成时（通过 `scheduler.is_finished()` API）
	1. 持续调用 `self.step()` 。`step()` 每次推进一轮调度与执行，并返回这一轮产生的输出（将在后面解释）
	2. 将获得的输出添加到 `outputs` 中，等待所有请求完成后输出

```python
class LLMEngine:
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
	    # (1-1): Tokenize
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # (1-2): 封装成 Sequence 对象
        seq = Sequence(prompt, sampling_params)
        
        # (1-3): 加入 Scheduler waiting queue
        self.scheduler.add(seq)
        
	def is_finished(self):
        return self.scheduler.is_finished()
        
	def generate(
	        self,
	        prompts: list[str] | list[list[int]],
	        sampling_params: SamplingParams | list[SamplingParams],
	        use_tqdm: bool = True,
	    ) -> list[str]:
	    
	    # (1) 将输入请求封装成 Sequence 对象，加入 Scheduler waiting queue
	    for prompt, sp in zip(prompts, sampling_params):
		    self.add_request(prompt, sp) # 见上方
		    
		outputs = {}
		# (2) 循环体判断条件：Scheduler 仍有请求没有完成
		while not self.is_finished():
			# (2-1)：持续调用 self.step() API
			output, num_tokens = self.step() # <- 【关键！】将会解释
			# (2-2): 将当前轮的输出加入 outputs
			for seq_id, token_ids in output:
				outputs[seq_id] = token_ids
				
		return outputs
```

在循环体内，我们看到 LLMEngine 调用了一个叫 `self.step()` 的函数，具体而言：
1. 调用 `scheduler.schedule()` API，决定：哪些 sequence 参与执行？这一轮属于 prefill 还是 decode batch？
2. 调用 `model_runner.run()` API，由 `is_prefill` bool 变量控制是执行 Prefill 还是 Decode 的一次前向计算
3. 调用 `scheduler.postprocess()` API，更新序列状态，例如追加生成 token，更新状态，维护 KV Cache 等

```python
class LLMEngine:
    def step(self):
	    # (2-1-1) 由 Scheduler 决定
		#     本轮要执行哪些请求
		#     本轮要执行 P 还是 D 的一次 iteration?
        seqs, is_prefill = self.scheduler.schedule()
        
        # 统计当前批处理的 token 数量，只是为了 benchmark 用
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        
        # （2-1-2） 执行这一轮模型的前向计算
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # （2-1-3） 更新序列状态，例如追加生成 token，更新状态，维护 KV Cache 等
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

总结：我们在 Scheduler 中需要实现的 API：
- `add()`：将请求添加到 waiting queue 中
- `is_finished()`：Scheduler 停止运行条件
- **`schedule()`：在每一轮决定：哪些 sequence 参与执行？这一轮属于 prefill 还是 decode batch？**
- `postprocess()` 在每一轮前向计算完成之后更新序列状态，例如追加生成 token，更新状态，维护 KV Cache 等
## 请求调度：Scheduler 类

> 本小节对应 Nano vLLM 的 [`class Scheduler`](https://github.com/GeeeekExplorer/nano-vllm/blob/812eb1c1e434576c0b7ae64d2cefb937aa80399d/nanovllm/engine/scheduler.py#L8)，点击链接跳转。