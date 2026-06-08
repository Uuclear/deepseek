# DeepSeek 部署快速入门（Windows / 跨平台）

> **定位**：从零到「模型能跑起来」——选路线、装环境、第一次对话。  
> **前置**：无需 AI 背景；会复制命令、会装软件即可。  
> **下一步**：跑通后读 [guide-01-inference-and-app-dev-zh.md](/part-06-practice/01-inference)

---

## 0. 先选一条路（30 秒决策）

| 路线 | 适合谁 | 成本 | 隐私 | 上手时间 |
|------|--------|------|------|----------|
| **A. 官方 API** | 最快验证想法、无 GPU | 按 token 付费 | 数据上云 | **10 分钟** |
| **B. Ollama 本地** | 有 8GB+ 显存或纯 CPU 试玩 | 免费 | 本地 | **30 分钟** |
| **C. vLLM / LM Studio** | 要高性能本地推理、多卡 | 硬件一次性 | 本地 | **1～2 小时** |
| **D. 云 GPU 租机** | 要跑 32B+ 或训练 | 按小时 | 看平台 | **2 小时+** |

**建议路径**：A 跑通业务逻辑 → B/C 本地私有化 → 再进入 [微调](/part-06-practice/02-finetuning) / [训练](/part-06-practice/04-rl-roadmap)。

---

## 1. 术语（部署篇）

**推理（Inference）**  
已经训练好的模型，根据输入生成输出。部署 = 把模型装到能对外提供服务的环境。

**API（Application Programming Interface）**  
程序调用模型的接口。DeepSeek 兼容 **OpenAI API 格式**，很多现有代码改 `base_url` 就能用。

**量化（Quantization）**  
把模型权重从 16 位压到 8/4 位，**显存占用大幅下降**，速度更快，精度略降。  
常见：`Q4_K_M`（Ollama）、AWQ/GPTQ（vLLM）。

**上下文（Context）**  
一次请求里「历史消息 + 当前问题」的总 token 数。越长占显存越多。

**DeepSeek 常见模型（个人部署）**

| 模型 | 体量 | 典型显存（FP16） | 量化后 | 特点 |
|------|------|------------------|--------|------|
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~14GB | ~5GB Q4 | 入门首选 |
| DeepSeek-R1-Distill-Qwen-14B | 14B | ~28GB | ~9GB Q4 | 单卡 24GB 可跑 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | ~64GB | ~20GB Q4 | 需 2×24GB 或租云 |
| deepseek-chat / deepseek-reasoner | 云端 | 无本地要求 | — | 官方 API |

> 671B 的 V3/R1 全量 **不适合个人本地部署**；用 API 或 Distill 小模型。

---

## 2. 路线 A：官方 API（最快）

### 2.1 注册与密钥

1. 打开 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册 → 创建 **API Key**（形如 `sk-...`）
3. 充值或领取试用额度

### 2.2 模型 ID

| 模型名 | 用途 |
|--------|------|
| `deepseek-chat` | 通用对话（对应 V3 能力） |
| `deepseek-reasoner` | 推理模式（对应 R1，带长 CoT） |

### 2.3 第一次调用（PowerShell + curl）

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的密钥"

curl https://api.deepseek.com/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:DEEPSEEK_API_KEY" `
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "用一句话解释什么是机器学习"}],
    "stream": false
  }'
```

### 2.4 Python 最小示例

```python
# pip install openai
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "1+1=?"}],
)
print(resp.choices[0].message.content)
```

**推理模型** 把 `model` 改成 `deepseek-reasoner`；返回里可能有 `reasoning_content` 字段（思考过程），见 [guide-01](/part-06-practice/01-inference)。

---

## 3. 路线 B：Ollama 本地（Windows 友好）

### 3.1 安装

1. 下载 [Ollama for Windows](https://ollama.com/download)
2. 安装后打开 PowerShell 验证：

```powershell
ollama --version
```

### 3.2 拉取 DeepSeek 相关模型

```powershell
# 通用对话（DeepSeek V3 系列，体积较大，按机器选择）
ollama pull deepseek-r1:7b

# 或 Distill 系（社区常见命名，以 ollama.com/library 当前列表为准）
ollama pull deepseek-r1:1.5b
```

> 模型名随 Ollama 库更新而变化；访问 https://ollama.com/search?q=deepseek 查看可用 tag。

### 3.3 命令行对话

```powershell
ollama run deepseek-r1:7b
```

### 3.4 本地 OpenAI 兼容 API

Ollama 默认暴露 `http://localhost:11434/v1`：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

resp = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

### 3.5 显存不够时

- 选更小 tag（1.5b / 7b）
- 关闭其他占 GPU 程序
- 无独显：Ollama 会用 CPU（慢但能用）

---

## 4. 路线 C：LM Studio（Windows GUI，零命令行）

适合：**不想敲命令、先体验本地模型**。

1. 下载 [LM Studio](https://lmstudio.ai/)
2. 搜索 `DeepSeek-R1-Distill` 或 `DeepSeek-R1`
3. 选 **Q4_K_M** 等等量化版本下载
4. 左侧「Local Server」→ Start Server（默认 OpenAI 兼容端口）
5. 在 Chat 页或外部程序连 `http://localhost:1234/v1`

---

## 5. 路线 C'：vLLM（Linux/WSL2，高性能）

Windows 原生 vLLM 支持有限；**推荐 WSL2 + Ubuntu** 或 Linux 云机。

### 5.1 WSL2 准备（Windows）

```powershell
wsl --install -d Ubuntu
```

在 Ubuntu 内：

```bash
# 需要 NVIDIA 驱动 + WSL CUDA
pip install vllm

# 单卡跑 7B
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192
```

### 5.2 多卡 32B 示例

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --tensor-parallel-size 2 \
  --max-model-len 16384
```

客户端同样用 OpenAI SDK，`base_url="http://localhost:8000/v1"`。

---

## 6. 路线 D：云 GPU 一键部署

| 平台 | 用途 |
|------|------|
| [AutoDL](https://www.autodl.com/) / [恒源云](https://gpushare.com/) | 国内租 A100/4090，按小时 |
| [RunPod](https://www.runpod.io/) | 海外 GPU |
| HuggingFace Spaces | 小 demo，免费档有限 |

典型流程：选镜像（PyTorch + CUDA）→ 挂载数据盘 → `pip install vllm` → `vllm serve ...` → 端口映射 → 本地用 API 访问。

**7B 蒸馏模型**：单卡 **4090 24GB** 足够 FP16；**32B** 建议 **2×4090** 或 A100 80GB。

---

## 7. 硬件对照表（怎么买/怎么租）

| 你的 GPU | 推荐本地模型 | 推荐方式 |
|----------|--------------|----------|
| 无独显 / 8GB | API 或 Ollama CPU | 路线 A / B |
| 12GB（3060 12G 等） | R1-Distill-7B Q4 | Ollama / LM Studio |
| 24GB（4090/3090） | 7B FP16 或 14B Q4 | Ollama / vLLM |
| 2×24GB | 32B Q4 或 7B 高 batch | vLLM TP=2 |
| 仅 CPU 32GB 内存 | 1.5B～3B Q4 | Ollama |

---

## 8. 部署验收清单（必须全部打勾）

```text
[ ] 能完成一次非流式对话并看到合理回复
[ ] 知道当前用的是 API 还是本地、模型名是什么
[ ] 能用 Python OpenAI SDK 发请求（改 base_url 即可）
[ ] 记录 max_tokens / 上下文限制，避免 OOM
[ ] （可选）流式输出能逐字显示
```

**验收脚本**（保存为 `smoke_test.py`）：

```python
import os, sys
from openai import OpenAI

BASE = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
KEY = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

client = OpenAI(base_url=BASE, api_key=KEY or "dummy")
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "回复 OK 两个字母"}],
    max_tokens=10,
)
text = r.choices[0].message.content.strip()
print("回复:", text)
sys.exit(0 if "OK" in text.upper() else 1)
```

API：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
python smoke_test.py
```

Ollama：

```powershell
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_MODEL="deepseek-r1:7b"
python smoke_test.py
```

---

## 9. 常见问题

**Q：API 报 401**  
A：检查 Key、是否 `Bearer ` 前缀、环境变量是否在当前终端生效。

**Q：Ollama 很慢**  
A：可能在用 CPU；看任务管理器 GPU 占用。换更小模型或 Q4 量化。

**Q：CUDA out of memory**  
A：减小 `max-model-len`、换 Q4、换小模型、关其他进程。

**Q：R1 和 Chat 区别？**  
A：API 上 `deepseek-reasoner` 会先「思考」再答，耗 token 多、数学题更强；`deepseek-chat` 更快更省。

**Q：和训练文档什么关系？**  
A：部署是 **用模型**；[训练解读](/part-07-theory/training-guide) 讲 **怎么做模型**。先会用，再学训。

---

## 10. 下一步

| 目标 | 文档 |
|------|------|
| 写应用、调 prompt、流式/RAG | [guide-01 应用开发](/part-06-practice/01-inference) |
| 用自己的数据改模型行为 | [guide-02 微调](/part-06-practice/02-finetuning) |
| 完整学习路线 | [guide-05 路线图](/part-06-practice/05-roadmap) |

---

## 示例文件

- [`examples/part-06-practice/00-smoke-test/smoke_test.py`](/examples/part-06-practice/00-smoke-test/smoke_test.py)
- [`examples/part-06-practice/00-smoke-test/.env.example`](/examples/part-06-practice/00-smoke-test/.env.example)

运行：`python examples/part-06-practice/00-smoke-test/smoke_test.py`
