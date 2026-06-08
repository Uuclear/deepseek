# 模型评估：Perplexity 与 Benchmark

> **前置知识**：[08-03 预训练](/part-08-llm-build/03-pretrain-ntp)、[Part 3 评估指标](/part-03-ml/05-evaluation-metrics)  
> **预计时间**：60 分钟  
> **本章产出**：会算困惑度并设计最小评测集

## Perplexity（困惑度）

语言模型在 held-out 文本上的平均负对数似然指数：

\[
\mathrm{PPL} = \exp\left(-\frac{1}{N}\sum_i \log P(x_i \mid x_{<i})\right)
\]

**越低越好**。迷你模型在极小语料上过拟合时 PPL 可极低，但不代表泛化。

## 计算片段（概念）

```python
# 伪代码：对验证集 batch 累计 cross_entropy，取 exp
```

在 `mini_gpt/train.py` 可增加验证循环：固定 `val` 字符序列，每 epoch 报告 val loss → PPL。

## 任务级 Benchmark

| 类型 | 例子 | 适用阶段 |
|------|------|----------|
| 知识 | MMLU 子集 | 预训练后 |
| 指令遵循 | 自建 50 题 Golden Set | SFT 后 |
| 推理 | GSM8K 小样 | 推理模型 |

个人项目优先 **自建 Golden Set**（见 [Part 6 guide-03](/part-06-practice/03-data-evaluation)）。

## 动手练习

1. 留 10% `corpus_zh_en.txt` 作验证，实现 val PPL 打印
2. 从 `sft_instructions.jsonl` 抽 10 条作「指令遵循」人工评分表
3. 说明 PPL 与聊天主观质量为何不总是一致

## 示例文件

- [`mini_gpt/train.py`](/examples/part-08-llm-build/mini_gpt/train.py)
- [Part 6 数据与评测](/part-06-practice/03-data-evaluation)

下一章：[08-08 衔接开源生态](/part-08-llm-build/08-scale-to-opensource)
