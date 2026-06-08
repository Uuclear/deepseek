# 数据集构造与清洗（JSONL 规范）

> **前置知识**：[Part 6 数据与评测](/part-06-practice/03-data-evaluation)  
> **预计时间**：60 分钟  
> **本章产出**：能编写符合规范的预训练 / SFT 数据文件

## 为什么用 JSONL

每行一个 JSON 对象，便于流式读取、追加、版本 diff。大规模训练管道（Megatron、Hugging Face datasets）普遍采用。

## 预训练语料行格式

```json
{"text": "单行或段落正文，已去 HTML 与敏感信息"}
```

清洗 checklist：

- 去重（精确或 MinHash）
- 过滤过短 / 乱码行
- 记录 `source`、`lang` 元数据（可选）
- 划分 train/val（如 99/1）

## SFT 语料

见 [`data/sft_instructions.jsonl`](/data/sft_instructions.jsonl)（Alpaca 三字段）：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

`input` 可为空字符串。

## 与本仓库数据

| 文件 | 格式 | 规模 |
|------|------|------|
| `corpus_zh_en.txt` | 纯文本 | ~50KB |
| `sft_instructions.jsonl` | Alpaca JSONL | 55 条 |

详见 [`data/README.md`](/data/README.md)。

## 动手练习

1. 手写 10 条原创 JSONL 指令，追加到本地副本
2. 写脚本统计：平均 `output` 长度、空 `input` 比例
3. 设计字段 `id`、`tags` 便于评测回溯

## 示例文件

- [`data/sft_instructions.jsonl`](/data/sft_instructions.jsonl)
- [`data/corpus_zh_en.txt`](/data/corpus_zh_en.txt)

下一章：[08-05 SFT 指令微调](/part-08-llm-build/05-sft-alpaca)
