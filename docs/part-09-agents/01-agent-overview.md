# Agent 是什么：感知-规划-行动-记忆

<LearningMeta
  prereq="<a href='/part-06-practice/01-inference'>Part 6 应用开发</a>"
  time="45 分钟"
  output="用四模块框架描述任意 AI Agent"
/>

## 定义

**Agent** = 在大模型之上，能 **感知环境 → 规划步骤 → 执行行动 → 更新记忆** 的系统。

单次 `chat.completions` 是「问答」；Agent 是「多步任务求解」。

## 四模块

| 模块 | 职责 | 例子 |
|------|------|------|
| 感知 | 读用户输入、工具返回、检索结果 | 用户问题 + RAG 片段 |
| 规划 | 分解任务、选工具、决定是否结束 | ReAct Thought |
| 行动 | 调 API、跑代码、写文件 | `get_weather("北京")` |
| 记忆 | 短期对话 + 长期知识库 | buffer + 向量库 |

## 与「只会聊天」的对比

```text
聊天：用户 → LLM → 文本
Agent：用户 → LLM ⇄ 工具/检索/记忆 → 多轮 → 最终结果
```

## 动手练习

1. 画一张你熟悉应用（如外卖 App）的四模块对照图
2. 运行 mock ReAct：`python examples/part-09-agents/01-react-agent/react_agent.py`
3. 列出 Agent 可能失败的三种情况（幻觉、工具错误、循环）

<ExampleBox
  title="Mock ReAct Agent"
  path="examples/part-09-agents/01-react-agent/react_agent.py"
  command="python examples/part-09-agents/01-react-agent/react_agent.py"
/>

---

下一章：[09-02 ReAct 与工具调用](/part-09-agents/02-react-tools)
