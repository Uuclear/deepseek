# MCP 协议入门与 Cursor Agent 生态

> **前置知识**：[09-03 Function Calling](/part-09-agents/03-function-calling)  
> **预计时间**：45 分钟  
> **本章产出**：理解 MCP 如何标准化 Agent 工具接入

## MCP 是什么

**Model Context Protocol** 让 Host（Cursor、Claude Desktop）以统一方式连接 **MCP Server**，获取 tools、resources、prompts。

类比：Function Calling 是「单次 API 工具」；MCP 是「可插拔工具服务器」。

## 核心组件

- **Server**：暴露能力（如浏览器、数据库、文件系统）
- **Client**：Agent 运行时调用 Server
- **Transport**：stdio 或 HTTP

## Cursor 中的实践

在 Cursor 设置中配置 MCP Server 后，Agent 可自动发现工具。阅读 [`data/sample_docs/05-mcp-intro.md`](/data/sample_docs/05-mcp-intro.md) 获取教程片段。

## 从本课程到 MCP

```text
Part 9 mock 工具 → Function Calling → 自研 Agent 循环 → MCP Server 封装
```

## 动手练习

1. 在 Cursor 文档中查看如何添加 MCP Server（记录一个你感兴趣的 Server）
2. 对比：把 `get_weather` 写成 Python 函数 vs MCP tool 的部署差异
3. 回顾 [Part 8](/part-08-llm-build/01-tokenizer-bpe)：训练模型与构建 Agent 如何配合

## 示例文件

- [`part-09-agents 总览`](/examples/part-09-agents/README.md)
- [`05-mcp-intro.md`](/data/sample_docs/05-mcp-intro.md)

---

**恭喜完成 Part 9！** 回到 [学习路线图](/roadmap) 查看 L8 能力自检，或深入 [Part 7 原理](/part-07-theory/training-guide)。
