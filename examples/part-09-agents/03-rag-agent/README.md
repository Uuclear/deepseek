# RAG Agent

基于 `data/sample_docs/` 的关键词检索；可选 DeepSeek API 生成回答。

```powershell
pip install openai python-dotenv
# 可选 Chroma 增强：pip install chromadb sentence-transformers
python examples/part-09-agents/03-rag-agent/rag_agent.py
```

无 API Key 时自动进入「仅检索」模式。
