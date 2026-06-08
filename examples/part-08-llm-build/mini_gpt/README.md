# mini_gpt — 迷你 GPT 预训练

字符级迷你 GPT，演示 **Attention + FFN + Next Token Prediction**。

```powershell
pip install torch
python examples/part-08-llm-build/mini_gpt/train.py --epochs 3
python examples/part-08-llm-build/mini_gpt/generate.py --prompt "深度学习"
```

- **GPU**：可选，CPU 数分钟即可跑通
- **数据**：`data/corpus_zh_en.txt`
- **产出**：`checkpoints/mini_gpt.pt`
