# SFT 指令微调（Alpaca 格式）

> **前置知识**：[08-03 预训练](/part-08-llm-build/03-pretrain-ntp)、[Part 6 微调指南](/part-06-practice/02-finetuning)  
> **预计时间**：90 分钟  
> **本章产出**：在迷你模型上完成一次 SFT 演示

## SFT 在流水线中的位置

```text
预训练（学语言） → SFT（学指令格式与对齐） → RLHF/DPO（可选，学偏好）
```

个人开发者主要接触 **SFT + LoRA**。

## Alpaca 模板

```text
指令:{instruction}
输入:{input}
回答:{output}
```

训练时把整段文本作为监督序列，模型学习在给定指令后生成 `回答:` 后的内容。

## 运行演示

```powershell
# 需先有 mini_gpt/checkpoints/mini_gpt.pt
python examples/part-08-llm-build/sft/train_sft.py
```

真实 7B 模型请用 LLaMA-Factory / Unsloth，数据格式与本章相同。

## 数据质量要点

- 覆盖目标任务分布
- 拒绝幻觉样本、格式错误样本
- 训练 / 验证拆分，防止过拟合单条模板

## 动手练习

1. 从 `sft_instructions.jsonl` 挑 5 条，手写「更优」回答并替换
2. 说明 SFT 与预训练损失有何异同（提示：数据分布、序列格式）
3. 阅读 [Part 6 guide-02](/part-06-practice/02-finetuning) 的 Alpaca JSON 示例并对照

## 示例文件

- [`sft/train_sft.py`](/examples/part-08-llm-build/sft/train_sft.py)
- [`data/sft_instructions.jsonl`](/data/sft_instructions.jsonl)

下一章：[08-06 LoRA / QLoRA](/part-08-llm-build/06-lora-qlora)
