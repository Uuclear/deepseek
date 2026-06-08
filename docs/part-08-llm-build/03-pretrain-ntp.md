# 预训练：Next Token Prediction

> **前置知识**：[08-02 迷你 GPT](/part-08-llm-build/02-mini-gpt-arch)  
> **预计时间**：90 分钟  
> **本章产出**：在 `corpus_zh_en.txt` 上跑通预训练并保存 checkpoint

## 训练目标

给定 token 序列 \(x_1,\dots,x_T\)，最大化：

\[
\sum_{t} \log P(x_t \mid x_{<t})
\]

实现上即：输入 `x[:-1]`，预测 `x[1:]`，交叉熵损失。

## 字符级 vs 子词级

本章示例为降低依赖，使用 **字符级** 词表（语料中每个 Unicode 字符一个 id）。工业模型用 BPE 词表 + 更大语料 + 多卡训练——流程相同，规模不同。

## 运行

```powershell
python examples/part-08-llm-build/mini_gpt/train.py --epochs 3
python examples/part-08-llm-build/mini_gpt/generate.py --prompt "机器学习"
```

| 环境 | 预计耗时 |
|------|----------|
| CPU | 2～5 分钟（3 epoch） |
| GPU | < 1 分钟 |

## 观察指标

- **训练 loss** 应缓慢下降（小模型 + 小语料会很快过拟合，属正常）
- **生成文本** 可能重复——说明需要更多数据/更大模型/更好分词

## 动手练习

1. 记录每个 epoch 的 loss，画简单折线图
2. 修改 `temperature` 看生成多样性变化
3. 对比：未训练随机权重 vs 训练后生成

## 示例文件

- [`mini_gpt/train.py`](/examples/part-08-llm-build/mini_gpt/train.py)
- [`mini_gpt/generate.py`](/examples/part-08-llm-build/mini_gpt/generate.py)

下一章：[08-04 数据集构造与清洗](/part-08-llm-build/04-dataset-jsonl)
