# ReAct Agent 模式

ReAct = **Re**asoning + **Act**ing：模型交替输出「思考」与「行动」。

## 循环

1. 观察用户问题与当前上下文
2. Thought：规划下一步
3. Action：选择工具与参数
4. Observation：执行工具，读回结果
5. 重复直到可给出 Final Answer

## 与 Function Calling 的关系

OpenAI 兼容 API 的 `tools` 参数把 Action 结构化为 JSON；ReAct 可用纯文本格式教学，再迁移到 function calling。
