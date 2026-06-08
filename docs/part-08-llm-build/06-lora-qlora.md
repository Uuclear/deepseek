# LoRA / QLoRA 原理与实战衔接

> **前置知识**：[08-05 SFT](/part-08-llm-build/05-sft-alpaca)、[Part 6 LoRA 微调](/part-06-practice/02-finetuning)  
> **预计时间**：75 分钟  
> **本章产出**：理解低秩适配与 4bit 量化训练如何节省显存

## LoRA 直觉

冻结原权重 \(W\)，学习低秩增量 \(\Delta W = BA\)，其中 \(B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}\)，\(r \ll d\)。

只在 attention / FFN 部分层插入 LoRA，即可显著改变行为，显存占用远低于全参微调。

## QLoRA

基座权重以 **4bit NF4** 存储，梯度在 FP16/BF16 LoRA 参数上更新。单卡 24GB 可微调 7B 级模型。

## 与本教程的关系

| 层级 | 教程位置 |
|------|----------|
| 迷你全参 SFT | `examples/part-08-llm-build/sft/` |
| 7B LoRA 实战 | [Part 6 guide-02](/part-06-practice/02-finetuning) |
| 原理与超参 | [Part 7 训练指南](/part-07-theory/training-guide) |

推荐路径：先跑通本章迷你 SFT → 直接用 Part 6 在 DeepSeek-Distill 上跑 LoRA。

## 动手练习

1. 列出 LoRA 的三个关键超参：`r`、`alpha`、`target_modules`
2. 说明为何 QLoRA 需要 `double quant` 与 `paged optimizers`（阅读 Unsloth 文档摘要即可）
3. 对比：全参 SFT 7B 与 QLoRA 7B 的显存量级

## 示例文件

- [Part 6 微调指南](/part-06-practice/02-finetuning)
- [`data/sft_instructions.jsonl`](/data/sft_instructions.jsonl) — 可转为 LLaMA-Factory 数据集

下一章：[08-07 模型评估](/part-08-llm-build/07-eval-perplexity)
