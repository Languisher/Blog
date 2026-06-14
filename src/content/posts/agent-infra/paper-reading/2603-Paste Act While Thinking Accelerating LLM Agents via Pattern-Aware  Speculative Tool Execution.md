---
title: "Paste: Act While Thinking: Accelerating LLM Agents via Pattern-Aware  Speculative Tool Execution"
published: 2026-06-10T15:09:08.360Z
description: ""
updated: ""
tags:
  - 论文阅读笔记
  - Agent-Infra
category: Agent-Infra
draft: true
pin: 0
toc: true
lang: ""
abbrlink: 2603-paste
---
## **目标**

PASTE 的目标是加速 LLM Agent 的端到端执行时间。它关注的不是单次 LLM inference，而是 Agent 中反复出现的“LLM thinking → tool execution → LLM thinking”串行循环。文章认为，tool execution 已经占到 Agent 总时间的很大比例，因此应该在 LLM 还在“思考”时，提前推测并执行可能会被调用的工具，从而把原本串行的工具等待时间隐藏掉。 

## **问题**

现代 Agent 通常遵循严格串行的 LLM-tool loop：LLM 先生成下一步动作和工具参数，系统再执行工具，工具结果返回后再进入下一轮 LLM。这个模型有天然依赖，所以工具调用不能简单并行。文章测量发现，tool execution 在 coding、deep research、scientific research 等任务中占据大量时间，约为总 latency 的 35% 到 61%，因此工具等待成为 Agent serving 的主要瓶颈。 

已有优化方法不太够。冷启动、Docker/Conda 环境复用、serverless workflow 优化只能减少一次性 startup overhead，但文章观察到工具初始化通常只占 tool latency 的小部分；普通 LLM serving 优化又主要优化 model inference，无法减少工具执行时间。更麻烦的是，Agent 的下一步工具调用由 LLM 在线生成，不像传统 serverless DAG 那样提前知道完整静态图，所以不能直接套用静态 DAG prefetch 或调度方法。 

真正困难在于：提前执行工具不仅要预测“下一个工具是什么”，还要预测“工具参数是什么”。工具参数可能是 URL、文件路径、搜索 query、代码片段等，很多来自上下文或前序工具输出。预测错了还可能产生副作用，例如错误安装依赖、修改环境、污染状态。因此 speculative tool execution 必须同时解决预测准确性、参数解析和安全隔离问题。 

## **Key insights / Observation**

第一，Agent 行为在语义层面看起来很多样，但在工具调用层面存在稳定的 control-flow pattern。比如 coding agent 中，file edit 后经常接 test / terminal execution；grep 后经常接 file_editor；research agent 中，search 后经常 fetch top URLs。这说明下一步工具调用不是完全随机的，而是有可挖掘的状态转移规律。 

第二，工具参数并不总是由 LLM 从零生成，很多参数可以从前序工具输出中直接派生。比如 web_fetch 的 URL 往往来自 search 返回 JSON 里的某个字段；file_editor 的文件名可能来自 grep 输出。文章特别指出，来自 prompt 或历史工具输出的参数对应的工具调用往往更耗时，因此提前做这类工具调用更有收益。 

第三，ReAct 形式上是串行的，但实际 workload 中存在 latent parallelism。比如 search 得到多个 URL 后，fetch 多个网页本来可以并行；代码报错后，多个相关文件也可以提前读取。PASTE 的核心判断是：不是强行改变 Agent 逻辑，而是在不影响主路径的前提下，把高概率、可安全执行的工具调用提前并行化。 

第四，speculation 不能和真实工具调用抢资源。错误预测不可避免，所以系统必须保证 misprediction 最坏只是浪费少量 slack resource，而不能拖慢 authoritative tool call。这一点是它和“盲目提前执行所有可能工具”的本质区别。 

## **解决的方法**

PASTE 提出 Pattern-Aware Speculative Tool Execution。它通过离线/在线分析 Agent 执行轨迹，挖掘稳定的工具调用模式，并用这些模式在运行时预测未来工具调用。当 LLM 还在生成下一步时，PASTE 就利用空闲资源提前执行预测工具；如果后续真实 Agent 调用了同一个工具和参数，就直接复用 speculative result，从而减少工具等待时间。 

它的核心抽象是 Pattern Tuple：`(context, prediction, function, probability)`。context 表示前序工具事件签名序列，只保留工具类型和执行状态，不保留具体自然语言 payload；prediction 表示预测的下一个工具；function 表示如何从历史工具输出中派生参数；probability 表示该模式在历史轨迹中的经验置信度。这个设计把 control flow 和 data flow 分开：先用工具序列识别“可能发生什么”，再用 value mapping function 解析“参数从哪里来”。 

在调度上，PASTE 把工具调用分成两类：authoritative invocation 和 speculative invocation。前者是 Agent 真实发出的工具调用， correctness-critical；后者是 PASTE 预测出来的 best-effort 工作。scheduler 严格优先真实调用，speculative jobs 只能使用 slack resources，并且一旦发生资源竞争就立即被抢占。这样 speculation 不会干扰正常 Agent 执行。 

如果真实工具调用到达时，发现已有匹配的 speculative job，PASTE 会执行 promotion：如果 speculative job 已完成，直接返回缓存结果；如果还在运行，就把它提升为 authoritative job，继续执行并提交结果。对于有副作用的工具，PASTE 使用用户定义的 speculation eligibility policy，只允许安全级别内的推测，比如 read-only full execution、partial warmup、dry-run 或 staging execution。最终，它以中间件/proxy 的形式接在 Agent 和 Tool backend 之间，记录 traces、预测工具调用、调度 speculative execution，同时保持原始工具 API 不变。