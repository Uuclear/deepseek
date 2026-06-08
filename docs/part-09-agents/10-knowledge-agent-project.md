# 实战项目：个人知识库 Agent（端到端）

<LearningMeta
  prereq="<a href='/part-09-agents/05-rag-agent'>Part 9-05 RAG</a>、<a href='/part-09-agents/09-agent-eval-safety'>Part 9-09 评测与安全</a>"
  time="2～3 小时"
  output="可查询 sample_docs 的个人知识库 Agent，支持 mock 与 DeepSeek API"
/>

本章把 Part 9 串成 **端到端项目**：加载原创文档 → 检索 →（可选）LLM 生成 → 带来源回答。

## 架构

```text
data/sample_docs/*.md
        ↓ 切分 chunk
   关键词索引（可升级为向量库）
        ↓
用户问题 → KnowledgeAgent.search() → top-k 片段
        ↓
   mock 模式：拼接片段返回
   API 模式：DeepSeek 根据片段生成答案
```

## 快速开始

```powershell
cd D:\github\deepseek
pip install python-dotenv openai
python examples/part-09-agents/06-knowledge-agent/knowledge_agent.py
```

可选 API 模式：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`。

<ExampleBox
  title="知识库 Agent"
  path="examples/part-09-agents/06-knowledge-agent/"
  command="python examples/part-09-agents/06-knowledge-agent/knowledge_agent.py --query 什么是 BPE 分词"
  download="/data/sample_docs/"
/>

## 数据准备

向 `data/sample_docs/` 放入你的笔记（Markdown/TXT）。仓库已含原创教程片段：

- `01-tokenizer-basics.md`
- `06-agent-eval-safety.md`（新增）
- `07-distributed-training.md`（新增）

**勿上传机密资料**；本地路径仅在运行时读取。

## 扩展方向

1. 换 `sentence-transformers` 做向量检索
2. 加 [Function Calling](/part-09-agents/03-function-calling) 查询日历、邮件
3. 用 [Part 9-09](/part-09-agents/09-agent-eval-safety) Golden Set 回归
4. 对接 [MCP](/part-09-agents/08-mcp-cursor) 供 Cursor 读取

## 项目检查清单

- [ ] `knowledge_agent.py` 在 mock 模式能回答 3 个测试问题
- [ ] 输出包含文档来源文件名
- [ ] 无 API Key 时优雅降级
- [ ] 可选：API 模式与 mock 对比同一问题

::: tip 与 Part 8 联动
知识库内容可包含你自己训练的 mini_gpt 笔记，形成「造模型 + 用 Agent 查询」闭环。
:::

---

上一章：[09-09 Agent 评测与安全](/part-09-agents/09-agent-eval-safety) · 下一章：[Part 10 GRPO 直觉](/part-10-advanced/01-grpo-rl-intuition)
