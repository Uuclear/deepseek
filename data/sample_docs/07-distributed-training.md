# 分布式训练入门（教程片段）

## DDP 数据并行

Distributed Data Parallel 在每张 GPU 上复制完整模型，各卡处理不同 mini-batch，反向传播后通过 All-Reduce 同步梯度并求平均，保证各卡权重一致。

## 启动方式

多卡训练常用 torchrun --nproc_per_node=N train.py。DataLoader 需配合 DistributedSampler 避免样本重复。

## 学习率缩放

全局 batch 增大时，常按线性规则放大学习率，并配合 warmup 稳定初期训练。

## 个人学习建议

先在单卡或 CPU 跑通 mini_gpt，再阅读开源训练脚本中的 DDP 配置即可，不必强求本地多卡环境。
