# 注意力机制

> **前置知识**：RNN 直觉  
> **预计时间**：60 分钟  
> **本章产出**：手算 tiny attention

Attention：Query 对 Key 算权重，加权求和 Value。

`softmax(QK^T / sqrt(d)) V` — Transformer 核心。

## 动手练习

修改 demo 里 Q 向量看权重变化

## 示例文件

- [`examples/part-05-nlp/03-attention/attention_demo.py`](/examples/part-05-nlp/03-attention/attention_demo.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-05-nlp/03-attention/attention_demo.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-05-nlp/04-transformer)
