# Agent 评测与安全

<LearningMeta
  prereq="<a href='/part-09-agents/05-rag-agent'>Part 9-05 RAG Agent</a>、<a href='/part-06-practice/03-data-evaluation'>Part 6 数据评测</a>"
  time="90 分钟"
  output="建立 Agent Golden Set，理解幻觉/工具滥用/沙箱防护"
/>

Agent 比单次对话更难评测：多步决策、工具副作用、检索质量都会引入失败模式。

## 评测维度

| 维度 | 测什么 | 指标示例 |
|------|--------|----------|
| 任务完成率 | 是否达成用户目标 | pass@1、步骤数 |
| 工具正确率 | 选对工具、参数合法 | tool accuracy |
|  groundedness | 回答是否基于检索 | 引用命中率 |
| 安全性 | 是否越权、泄露 | 红队用例通过率 |
| 成本/延迟 | 可部署性 | token、轮次 |

本仓库提供 [`data/eval_questions.jsonl`](/data/eval_questions.jsonl) 作为 **LLM + Agent 共用 Golden Set**。

## 幻觉（Hallucination）

**表现**：编造事实、假引用、错误工具结果仍自信输出。

**缓解**：

1. RAG 强制「仅根据上下文回答」
2. 要求引用 `doc_id`
3. 低置信度时回复「资料不足」
4. 用 Golden Set 回归测试

```python
# 评测脚本会检查回答是否包含期望关键词
expected_keywords = ["ReAct", "Thought", "Action"]
```

## 工具滥用

**表现**：无限循环调工具、危险命令、越权读写。

**缓解**：

- **白名单工具**：只允许注册过的 function
- **步数上限**：`max_steps=10`
- **人工确认**：写文件、发邮件前二次确认
- **沙箱执行**：见下文

## 沙箱（Sandbox）

代码执行类 Agent 应在隔离环境运行：

```text
用户请求 → Agent 生成代码 → 沙箱 subprocess（超时、禁网、临时目录）
                              ↓
                         仅返回 stdout/stderr
```

::: danger 切勿
在生产环境直接 `exec()` 模型生成的 shell 命令。教程示例使用 mock 或只读检索。
:::

## 动手：跑评测脚本

```powershell
cd D:\github\deepseek
python examples/part-10-advanced/eval_runner/eval_llm.py
```

脚本读取 Golden Set，对 mock 检索结果做关键词匹配（无需 API）。

<ExampleBox
  title="Golden Set"
  path="data/eval_questions.jsonl"
  download="/data/eval_questions.jsonl"
  command="python examples/part-10-advanced/eval_runner/eval_llm.py"
/>

## 自检清单

- [ ] 为每个工具写 3 条失败用例（参数缺失、类型错误、超时）
- [ ] RAG Agent 回答附带来源 chunk id
- [ ] 记录每轮 token 与工具调用次数

---

上一章：[09-08 MCP 与 Cursor](/part-09-agents/08-mcp-cursor) · 下一章：[09-10 个人知识库 Agent](/part-09-agents/10-knowledge-agent-project)
