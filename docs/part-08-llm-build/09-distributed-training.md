# 分布式训练入门（DDP 概念）

<LearningMeta
  prereq="<a href='/part-08-llm-build/03-pretrain-ntp'>Part 8-03 预训练 NTP</a>、<a href='/part-04-dl/05-training-loop'>Part 4 训练循环</a>"
  time="60 分钟（阅读为主）"
  output="能解释 DDP 数据并行流程，知道何时需要多卡训练"
/>

> 个人学习者**不必亲自跑多卡**，但理解分布式训练有助于阅读开源训练脚本与论文。

## 为什么需要分布式

单卡显存有限；大模型预训练需要：

- **数据并行（DP / DDP）**：多卡各持一份模型副本，分 batch 训练，梯度同步
- **模型并行**：单层太大时切分到多卡（张量并行、流水线并行）
- **混合并行**：工业级训练常组合使用

本章聚焦最易理解的 **DDP（Distributed Data Parallel）**。

## DDP 直觉

```text
1. 每张 GPU 加载相同模型权重
2. 各卡处理不同 mini-batch 子集
3. 前向 + 反向得到本地梯度
4. All-Reduce 同步梯度（求平均）
5. 各卡用相同梯度更新 → 权重保持一致
```

PyTorch 伪代码：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组（多进程，每进程一张卡）
dist.init_process_group(backend="nccl")
model = MiniGPT(...).to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## 与单卡训练的差异

| 维度 | 单卡 | DDP |
|------|------|-----|
| 启动 | `python train.py` | `torchrun --nproc_per_node=4 train.py` |
| DataLoader | 普通 shuffle | `DistributedSampler` 保证不重复 |
| 学习率 | 基准 lr | 常按全局 batch 线性缩放 |
| 保存 | 直接 `state_dict` | 取 `model.module.state_dict()` |

::: tip 个人学习建议
先用 `mini_gpt/train.py` 在 CPU/单卡跑通；读懂 DDP 概念后，再浏览 Hugging Face `Trainer` 或 DeepSpeed 配置即可。
:::

## 常见问题

1. **为什么 loss 和单卡不一样？** 全局 batch 变大，有效学习率变化，需调 lr 或 warmup。
2. **Windows 能跑 DDP 吗？** 官方推荐 Linux + NCCL；Windows 可用 gloo，但多卡环境较少，本教程不要求实操。
3. **和 DeepSeek 的关系？** 大规模预训练依赖集群 DDP + 模型并行；推理侧用 vLLM 等（见 [Part 10-04](/part-10-advanced/04-vllm-deployment)）。

## 动手练习

1. 阅读 `torchrun` 官方文档，写出 4 卡启动命令
2. 解释 `DistributedSampler` 如何避免各卡读到重复样本
3. 对比 [Part 8-10 量化导出](/part-08-llm-build/10-quantization-export)：训练用 FP32/BF16，部署为何需要 INT4/GGUF

<ExampleBox
  title="相关示例"
  path="examples/part-08-llm-build/mini_gpt/"
  command="python examples/part-08-llm-build/mini_gpt/train.py --epochs 3"
/>

---

上一章：[08-08 衔接开源生态](/part-08-llm-build/08-scale-to-opensource) · 下一章：[08-10 模型量化与导出](/part-08-llm-build/10-quantization-export)
