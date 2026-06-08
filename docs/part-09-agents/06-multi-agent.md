# 多 Agent 协作（Supervisor + Worker）

> **前置知识**：[09-05 RAG Agent](/part-09-agents/05-rag-agent)  
> **预计时间**：75 分钟  
> **本章产出**：理解任务分解与角色分工

## 为什么需要多 Agent

复杂任务（调研 + 写作 + 审校）单 Agent 易超长上下文、角色混乱。拆成 **Supervisor** 调度多个 **Worker** 更清晰。

## Supervisor 模式

```text
Supervisor：读目标 → 生成计划 → 派发给 Researcher / Writer / Critic
Worker：执行子任务 → 返回结构化结果
Supervisor：合并 → 决定迭代或结束
```

## 示例

```powershell
python examples/part-09-agents/04-multi-agent/supervisor_demo.py
```

当前为规则演示；可替换为 LLM 驱动计划与派发（LangGraph、AutoGen 等框架）。

## 框架选型（了解即可）

| 框架 | 特点 |
|------|------|
| LangGraph | 图状态机，适合可控流程 |
| CrewAI | 角色配置声明式 |
| 自研 while 循环 | 依赖最少，教学首选 |

## 动手练习

1. 增加 `Critic` worker，对 Writer 输出打分
2. 画 sequence 图：Supervisor 与两个 Worker 的消息流
3. 说明多 Agent 相对单 Agent 的成本（token、延迟）

## 示例文件

- [`supervisor_demo.py`](/examples/part-09-agents/04-multi-agent/supervisor_demo.py)

下一章：[09-07 Agent 记忆](/part-09-agents/07-agent-memory)
