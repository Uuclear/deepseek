# Transformer 结构

> **前置知识**：Attention  
> **预计时间**：75 分钟  
> **本章产出**：认识 Encoder 块

多层：**Self-Attention** → Add&Norm → **FFN** → Add&Norm。

`nn.MultiheadAttention` 已实现多头注意力。

## 动手练习

运行 demo 打印输出 shape

## 示例文件

- [`examples/part-05-nlp/04-transformer-blocks/demo.py`](/examples/part-05-nlp/04-transformer-blocks/demo.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-05-nlp/04-transformer-blocks/demo.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-05-nlp/05-pretraining)
