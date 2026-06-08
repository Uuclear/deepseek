# Part 8 — 从零构建大模型示例

## 依赖

```powershell
cd D:\github\deepseek
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r examples\requirements.txt
pip install tokenizers
```

| 脚本 | GPU | 说明 |
|------|-----|------|
| `train_tokenizer.py` | 否 | 在 `data/corpus_zh_en.txt` 上训练 BPE |
| `mini_gpt/train.py` | 可选 | CPU 可跑极小 GPT 预训练 |
| `mini_gpt/generate.py` | 可选 | 加载 checkpoint 生成文本 |
| `sft/train_sft.py` | 可选 | 迷你模型 SFT 演示 |

## 快速开始

```powershell
# 1. 训练分词器
python examples/part-08-llm-build/train_tokenizer.py

# 2. 预训练迷你 GPT（约 2～5 分钟 CPU）
python examples/part-08-llm-build/mini_gpt/train.py

# 3. 生成
python examples/part-08-llm-build/mini_gpt/generate.py --prompt "机器学习"

# 4. SFT 演示
python examples/part-08-llm-build/sft/train_sft.py
```

输出目录：`examples/part-08-llm-build/output/`、`mini_gpt/checkpoints/`。
