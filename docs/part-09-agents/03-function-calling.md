# Function Calling 与 OpenAI 兼容 API（DeepSeek）

> **前置知识**：[09-02 ReAct](/part-09-agents/02-react-tools)、[Part 6 推理](/part-06-practice/01-inference)  
> **预计时间**：75 分钟  
> **本章产出**：用 DeepSeek API 完成一次 tool call 往返

## 流程

1. 定义 `tools` JSON schema（见 [`data/agent_tools.json`](/data/agent_tools.json)）
2. `chat.completions.create(..., tools=tools)`
3. 若 `message.tool_calls` 非空 → 本地执行 → `role: tool` 回传
4. 再次请求得到自然语言答案

DeepSeek API 兼容 OpenAI SDK，仅需修改 `base_url`。

## 配置

```powershell
copy examples\part-09-agents\02-function-calling\.env.example examples\part-09-agents\02-function-calling\.env
python examples/part-09-agents/02-function-calling/function_calling_demo.py
```

**需要**：`DEEPSEEK_API_KEY`（无 GPU 要求）

## Schema 设计要点

- `name` 动词短语、`description` 写清何时调用
- 参数 `required` 最小化
- 教学项目用 mock 函数，避免真实副作用

## 动手练习

1. 为 `search_docs` 写 schema 并在 demo 中接入
2. 记录一次完整 messages 列表（含 tool 消息）
3. 说明 `tool_call_id` 为何必须对齐

## 示例文件

- [`function_calling_demo.py`](/examples/part-09-agents/02-function-calling/function_calling_demo.py)
- [`data/agent_tools.json`](/data/agent_tools.json)

下一章：[09-04 构建单 Agent](/part-09-agents/04-single-agent)
