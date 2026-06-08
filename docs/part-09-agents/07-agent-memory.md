# Agent 记忆：短期 / 长期 / 向量库

> **前置知识**：[09-06 多 Agent](/part-09-agents/06-multi-agent)  
> **预计时间**：60 分钟  
> **本章产出**：区分三种记忆并跑通 JSON 长期记忆示例

## 三类记忆

| 类型 | 存储 | 生命周期 | 用途 |
|------|------|----------|------|
| 短期 | `messages` 列表 | 单会话 | 多轮对话连贯 |
| 长期 | DB / JSON / 文件 | 跨会话 | 用户偏好、事实 |
| 向量 | Chroma / FAISS | 跨会话可检索 | 大量非结构化记忆 |

## 示例

```powershell
python examples/part-09-agents/05-memory-agent/memory_agent.py
```

演示：对话 buffer + `memory_store.json` 键值长期记忆。

## 与 RAG 的边界

- **RAG**：外部知识库，人人相同
- **记忆**：用户特定、Agent 运行中写入

生产系统常两者结合：RAG 查文档，记忆查用户历史。

## 动手练习

1. 扩展 `remember()` 支持删除与列表
2. 设计「记忆摘要」：会话结束前让 LLM 压缩 buffer 写入长期
3. 说明向量记忆何时优于键值记忆

## 示例文件

- [`memory_agent.py`](/examples/part-09-agents/05-memory-agent/memory_agent.py)

下一章：[09-08 MCP 入门](/part-09-agents/08-mcp-cursor)
