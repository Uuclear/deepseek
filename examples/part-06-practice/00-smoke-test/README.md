# smoke_test — 部署验收

验证 API 或本地 OpenAI 兼容端点是否可用。

## 依赖

```powershell
pip install openai python-dotenv
```

## 运行（官方 API）

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
python smoke_test.py
```

## 运行（Ollama）

```powershell
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_MODEL="deepseek-r1:7b"
python smoke_test.py
```

预期：打印包含 `OK` 的回复，退出码 0。
