# sft — 指令微调演示

在 `data/sft_instructions.jsonl`（Alpaca 格式）上对迷你 GPT 做短程 SFT。

```powershell
# 需先完成 mini_gpt 预训练
python examples/part-08-llm-build/sft/train_sft.py
```

教学说明：真实场景应使用 tokenizer + 更大基座 + LoRA；详见 Part 8 §08-05、§08-06 与 Part 6 [微调指南](/part-06-practice/02-finetuning)。
