# BPE / WordPiece 分词器从零训练

<LearningMeta
  prereq="<a href='/part-05-nlp/01-tokenization-embedding'>Part 5 分词与 Embedding</a>"
  time="90 分钟"
  output="在真实小语料上训练并保存 BPE 分词器"
/>

## 为什么从分词开始

大模型的输入不是「字符」或「单词」，而是 **token id 序列**。工业级模型使用子词算法（BPE、WordPiece、Unigram）在语料上统计合并，平衡词表大小与 OOV（未登录词）问题。

## BPE 直觉

1. 初始化：字符级词表 + 特殊符号 `[UNK]`
2. 统计相邻符号对频率，合并最高频的一对
3. 重复直到达到目标 `vocab_size`

WordPiece 与 BPE 类似，合并准则略有不同；Hugging Face `tokenizers` 库同时支持。

## 动手：训练教程语料分词器

我们使用仓库原创语料 [`data/corpus_zh_en.txt`](/data/corpus_zh_en.txt)（中英文混合，约 100KB）。

```powershell
pip install tokenizers
python examples/part-08-llm-build/train_tokenizer.py
```

产出：`examples/part-08-llm-build/output/bpe_tokenizer.json`

## 与 Part 5 的衔接

Part 5 用正则做**教学级**分词；本章进入**可部署**的子词训练流程，为下一章迷你 GPT 准备词表。

## 动手练习

1. 将 `vocab_size` 改为 500 / 1200，观察 token 切分差异
2. 对一句纯英文、一句纯中文、一句中英混合分别 `encode`，记录 tokens
3. 思考：为什么中文语料需要足够规模才能学到合理的子词？

<ExampleBox
  title="本章示例"
  path="examples/part-08-llm-build/train_tokenizer.py"
  command="pip install tokenizers&#10;python examples/part-08-llm-build/train_tokenizer.py"
  download="/data/corpus_zh_en.txt"
/>

---

下一章：[08-02 迷你 GPT 架构](/part-08-llm-build/02-mini-gpt-arch)
