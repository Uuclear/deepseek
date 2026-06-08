# ReAct 模式与工具调用

> **前置知识**：[09-01 Agent 概述](/part-09-agents/01-agent-overview)  
> **预计时间**：60 分钟  
> **本章产出**：理解 Thought / Action / Observation 循环

## ReAct 论文核心

让模型 **交错生成推理轨迹与工具调用**，用 Observation 把外部世界反馈写回上下文，减少纯幻觉。

典型格式：

```text
Thought: 需要查天气
Action: get_weather(city="北京")
Observation: 晴 25°C
Thought: 可以回答用户了
Final Answer: 北京今天晴，25°C。
```

## 实现方式

| 方式 | 优点 | 缺点 |
|------|------|------|
| 纯文本 ReAct | 教学清晰、模型无关 | 解析脆弱 |
| Function Calling | 结构化 JSON、官方 SDK | 依赖 API |

## 运行示例

```powershell
python examples/part-09-agents/01-react-agent/react_agent.py
```

## 动手练习

1. 给 mock Agent 增加 `search_docs` 工具分支
2. 设计 max_steps 防止死循环
3. 对比 ReAct 与「一次性把工具说明塞进 system prompt」的差异

## 示例文件

- [`react_agent.py`](/examples/part-09-agents/01-react-agent/react_agent.py)

下一章：[09-03 Function Calling](/part-09-agents/03-function-calling)
