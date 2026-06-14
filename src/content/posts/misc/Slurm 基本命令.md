---
title: Slurm 基本命令
published: 2026-06-11T09:11:12.214Z
description: ""
updated: ""
tags:
  - Slurm
category: Misc.
draft: false
pin: 0
toc: true
lang: ""
abbrlink: slurm-guide
---
Slurm 是一个具备容错能力和可扩展性的集群管理与作业调度系统，，用于管理计算资源分配、执行作业以及解决资源争用问题。

## 作为用户，我们需要知道……

### 如何启动一项作业：`srun` 和 `sbatch`

#### 【方法 1】：使用 `srun`

`srun` 用于直接启动一个任务，常用于交互式调试。

例如：

```sh
srun -p GPU --gres=gpu:1 -c 4 --mem=20G --pty bash
```

进入分配到的计算节点后，再运行：

```sh
echo $CUDA_VISIBLE_DEVICES # 验证只看到了 GPU 1
python your_script.py
```

**应用场景。** 适合临时测试、交互式调试、检查环境、GPU、路径是否正确


#### 【方法2】：使用 `sbatch`


`sbatch` 用于提交批处理脚本以供稍后执行。任务会进入队列，等资源可用后自动运行。例如：

```sh
sbatch example.sh
```

这个脚本通常包括：
- 资源申请：CPU、GPU、内存、最长运行时间等
- 日志配置：标准输出和标准错误保存到哪里
- 环境初始化：例如加载模块、激活 conda 环境
- 实际任务命令：例如运行 Python 脚本
- 可选清理逻辑：例如删除临时文件、关闭后台进程

示例脚本 `example.sh` 内容：

```sh
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --export=ALL

echo "Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Current directory: $(pwd)"
echo "Start time: $(date)"

# source ~/.bashrc
# conda activate your_env_name

python your_script.py

echo "Job finished"
echo "End time: $(date)"
```

CPU 任务参数：

| 参数           | 示例                              | 是否通常必写 | 作用               |
| ------------ | ------------------------------- | ------ | ---------------- |
| **`-p`**     | **`#SBATCH -p CPU`**            | **是**  | **指定提交到 CPU 分区** |
| **`-c`**     | **`#SBATCH -c 4`**              | **是**  | **申请 CPU 核数**    |
| **`--mem`**  | **`#SBATCH --mem=20G`**         | **是**  | **申请内存**         |
| **`--time`**     | **`#SBATCH --time=24:00:00`**       | **是**      | **设置最大运行时间**         |
| `--job-name` | `#SBATCH --job-name=my_job`     | 建议     | 设置任务名称           |
| `--output`   | `#SBATCH --output=slurm-%j.out` | 建议     | 保存标准输出日志         |
| `--error`    | `#SBATCH --error=slurm-%j.err`  | 建议     | 保存标准错误日志         |
| `--export`   | `#SBATCH --export=ALL`          | 可选     | 继承当前环境变量         |
GPU 任务参数：

| 参数           | 示例                              | 是否通常必写 | 作用                       |
| ------------ | ------------------------------- | ------ | ------------------------ |
| **`-p`**     | **`#SBATCH -p GPU`**            | **是**  | **指定提交到 GPU 分区**         |
| **`--gres`** | **`#SBATCH --gres=gpu:1`**      | **是**  | **申请 GPU 资源**            |
| **`-c`**     | **`#SBATCH -c 20`**             | **是**  | **申请 CPU 核数（数据加载、预处理等）** |
| **`--mem`**  | **`#SBATCH --mem=50G`**         | **是**  | **申请内存**                 |
| `--time`     | `#SBATCH --time=48:00:00`       | 是      | 设置最大运行时间                 |
| `--job-name` | `#SBATCH --job-name=my_job`     | 建议     | 设置任务名称                   |
| `--output`   | `#SBATCH --output=slurm-%j.out` | 建议     | 保存标准输出日志                 |
| `--error`    | `#SBATCH --error=slurm-%j.err`  | 建议     | 保存标准错误日志                 |
| `--export`   | `#SBATCH --export=ALL`          | 可选     | 继承当前环境变量                 |
### 如何查看和管理作业

查看自己的作业：

```sh
squeue -u $USER
```

取消作业：

```sh
scancel JOB_ID
```

查看某个作业的详细信息：

```sh
scontrol show job JOB_ID
```
## 参考资料

- [Slurm全攻略：高效玩转HPC](https://zhuanlan.zhihu.com/p/1900465941864748441)