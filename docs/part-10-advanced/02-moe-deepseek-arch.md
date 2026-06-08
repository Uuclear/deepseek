# MoE 与 DeepSeek 架构要点

<LearningMeta
  prereq="<a href='/part-07-theory/v3-architecture'>Part 7 V3 架构</a>、<a href='/part-05-nlp/04-transformer'>Part 5 Transformer</a>"
  time="75 分钟"
  output="理解 MoE 路由、负载均衡、MLA 与 DeepSeek-V3 设计取舍"
/>

**MoE（Mixture of Experts）** 让模型「总参数量大、激活参数量小」— 推理成本接近小模型，容量接近大模型。

## 稠密 vs MoE

| | 稠密 Transformer | MoE |
|--|------------------|-----|
| 每层 FFN | 全部神经元参与 | 只激活 top-k 个专家 |
| 参数量 | 全部参与计算 | 总参数 ↑，激活参数 ≈ 稠密 |
| 挑战 | 显存随层数线性涨 | 路由、负载均衡、通信 |

## 路由直觉

```text
hidden_state → Router（线性层 + softmax）
            → 选 top-2 专家
            → 加权求和专家输出
```

**负载均衡损失**：避免所有 token 都涌向同一专家（否则其余专家浪费）。

## DeepSeek 相关要点

- **MLA（Multi-head Latent Attention）**：压缩 KV，减推理显存（[Part 7 详解](/part-07-theory/v3-architecture)）
- **DeepSeek-V3**：大规模 MoE 预训练 + 长上下文优化
- **DeepSeek-R1**：在后训练阶段强化推理链能力

个人 mini_gpt 是稠密小模型；理解 MoE 有助于阅读官方技术报告与 HF 配置。

## 动手练习

1. 画一张「8 专家 top-2」示意图，标出激活路径
2. 解释为何 MoE 训练需要 **专家并行**
3. 阅读 `config.json` 中 `num_experts`、`num_experts_per_tok` 字段（任意 MoE 开源模型）

---

上一章：[10-01 GRPO 直觉](/part-10-advanced/01-grpo-rl-intuition) · 下一章：[10-03 蒸馏实战](/part-10-advanced/03-distillation-practice)
