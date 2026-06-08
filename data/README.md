# 数据集说明

本目录提供教程用的**原创**小数据集，可自由用于学习与实验。

| 文件 / 目录 | 用途 | 关联章节 |
|-------------|------|----------|
| `corpus_zh_en.txt` | 中英文混合语料（~100KB），训练 BPE 分词器 | Part 8 §08-01 |
| `sft_instructions.jsonl` | 80 条 Alpaca 格式中文指令数据 | Part 8 §08-05 |
| `eval_questions.jsonl` | LLM / Agent Golden Set 评测题 | Part 9 §09-09 |
| `iris.csv` | 经典 Iris 分类数据 | Part 3 |
| `titanic_sample.csv` | Titanic 子集 | Part 3 |
| `sample_corpus.txt` | 早期示例语料（较短） | Part 5 |
| `agent_tools.json` | Function Calling 工具 schema 示例 | Part 9 §09-03 |
| `sample_docs/` | 9 篇 AI 教程片段（Markdown/TXT） | Part 9 RAG / 知识库 Agent |

## 使用方式

构建站点后可通过 `/data/...` 下载；本地开发时直接读取仓库根目录 `data/`：

```powershell
# 从仓库根目录
python examples/part-08-llm-build/train_tokenizer.py
python examples/part-10-advanced/eval_runner/eval_llm.py
```

## 许可

内容为教程作者原创，MIT 许可与主仓库一致。
