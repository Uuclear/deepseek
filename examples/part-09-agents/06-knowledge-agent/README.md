# 06 个人知识库 Agent

端到端示例：检索 `data/sample_docs/` 中的原创教程文档，mock 或 DeepSeek API 模式回答。

## 运行

```powershell
cd D:\github\deepseek
pip install python-dotenv openai
python examples/part-09-agents/06-knowledge-agent/knowledge_agent.py --query "什么是 ReAct Agent"
```

## API 模式（可选）

```powershell
copy examples\part-09-agents\06-knowledge-agent\.env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
```

## 文档

[Part 9-10 实战项目](/part-09-agents/10-knowledge-agent-project)
