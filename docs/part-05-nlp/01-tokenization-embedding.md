# 分词与 Embedding

> **前置知识**：PyTorch 基础  
> **预计时间**：60 分钟  
> **本章产出**：文本→token→向量

**分词**把文本切成 token（字/词/子词 BPE）。

**Embedding** 把离散 id 映射为稠密向量，语义相近则向量接近。

## 动手练习

统计 sample_corpus 唯一词数

## 示例文件

- [`examples/part-05-nlp/01-tokenizer/simple_tokenizer.py`](/examples/part-05-nlp/01-tokenizer/simple_tokenizer.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-05-nlp/01-tokenizer/simple_tokenizer.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-05-nlp/02-rnn-intuition)
