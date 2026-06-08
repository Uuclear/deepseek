# 构建单 Agent（Python + OpenAI SDK）

> **前置知识**：[09-03 Function Calling](/part-09-agents/03-function-calling)  
> **预计时间**：90 分钟  
> **本章产出**：一个可扩展的单 Agent 循环骨架

## 最小 Agent 类结构

```text
class Agent:
    tools: dict[str, Callable]
    messages: list

    def step(user_input) -> str:
        append user message
        call LLM with tools
        while tool_calls:
            execute locally
            append tool results
            call LLM again
        return assistant text
```

`function_calling_demo.py` 即单步版；可扩展为 `while` 多轮直到无 tool_calls。

## 工程清单

- `.env` 管理密钥（勿提交 Git）
- 超时与重试（网络/API 限流）
- 日志：每步 Thought / 工具入参出参
- 单元测试：mock LLM 返回固定 `tool_calls`

## 与 Part 6 的关系

[guide-01 应用开发](/part-06-practice/01-inference) 的 `llm_client.py` 可复用为 Agent 的 LLM 层。

## 动手练习

1. 把 demo 改成函数 `run_agent(query: str) -> str`
2. 增加 system prompt：「你是教程助手，必须先查工具再回答」
3. 限制工具调用次数 `max_tool_rounds=3`

## 示例文件

- [`02-function-calling/`](/examples/part-09-agents/02-function-calling/README.md)
- [`part-06 01-llm-client`](/examples/part-06-practice/01-llm-client/README.md)

下一章：[09-05 RAG Agent](/part-09-agents/05-rag-agent)
