# RAG Agent：检索 + 推理 + 引用

> **前置知识**：[09-04 单 Agent](/part-09-agents/04-single-agent)、[Part 6 RAG 提及](/part-06-practice/01-inference)  
> **预计时间**：90 分钟  
> **本章产出**：能检索本地文档并生成带引用的回答

## RAG Agent 流水线

```text
文档库 → 切分 chunk → 嵌入/索引 → 用户提问 → 检索 top-k → 拼 prompt → LLM → 带出处回答
```

## 本仓库示例

- 文档：[`data/sample_docs/`](/data/sample_docs/)（5 篇原创教程片段）
- 脚本：[`rag_agent.py`](/examples/part-09-agents/03-rag-agent/rag_agent.py)

```powershell
python examples/part-09-agents/03-rag-agent/rag_agent.py
```

无 API Key 时输出检索片段；有 Key 时调用 DeepSeek 生成。

## 进阶：Chroma

可选安装 `chromadb` + `sentence-transformers` 做向量检索，替换示例中的关键词匹配。

## 动手练习

1. 往 `sample_docs/` 增加一篇你写的 Markdown 笔记
2. 修改 prompt 强制「必须引用 [文件名]」
3. 对比 top_k=1 与 top_k=5 的答案质量

## 示例文件

- [`03-rag-agent/`](/examples/part-09-agents/03-rag-agent/README.md)
- [`data/sample_docs/`](/data/sample_docs/)

下一章：[09-06 多 Agent 协作](/part-09-agents/06-multi-agent)
