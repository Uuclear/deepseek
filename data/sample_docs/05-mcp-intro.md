# MCP 协议入门

Model Context Protocol（MCP）让 AI 应用以标准方式连接外部工具与数据源。

## 核心概念

- **Server**：暴露 tools / resources / prompts
- **Client**：Agent 或 IDE（如 Cursor）调用 Server
- **Transport**：stdio 或 HTTP

## 与 Agent 开发的关系

学会 function calling 后，MCP 是把同一能力「产品化、可插拔」的下一步；Cursor 内置多个 MCP Server 供 Agent 使用。
