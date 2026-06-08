# 推理调用与应用开发入门

> **前置**：[guide-00 部署](/part-06-practice/00-deployment) 已跑通 smoke test。  
> **目标**：会用 API 写小应用、理解参数、处理 R1 推理输出、为微调/RAG 打基础。

---

## 1. 核心概念

**Chat Completions**  
多轮对话接口：消息列表 `messages`，每条有 `role`（`system` / `user` / `assistant`）和 `content`。

**Token**  
计费与长度单位。中文约 1～2 字/token。`max_tokens` 限制 **模型生成** 的上限，不含输入。

**Temperature（温度）**  
0～2，越高越随机。  
- 代码/数学：**0～0.3**  
- 创意写作：**0.7～1.0**  
- R1 官方评测常用：**0.6**

**Top-p（核采样）**  
从概率累计 top-p 的 token 里采样。R1 评测常用 **0.95**。

**Streaming（流式）**  
边生成边返回，适合聊天 UI；需处理 SSE 事件流。

**System Prompt（系统提示）**  
设定角色与规则，一般不展示给终端用户。

**Reasoning Content（推理内容）**  
`deepseek-reasoner` 或本地 R1 模型可能单独返回思考过程；展示给用户时要区分「思考」与「最终答案」。

---

## 2. OpenAI 兼容调用模板

### 2.1 多轮对话

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
)

messages = [
    {"role": "system", "content": "你是简洁的技术助手，回答用中文。"},
    {"role": "user", "content": "什么是 LoRA？"},
]

resp = client.chat.completions.create(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    messages=messages,
    temperature=0.3,
    max_tokens=1024,
)
reply = resp.choices[0].message.content
messages.append({"role": "assistant", "content": reply})

# 第二轮
messages.append({"role": "user", "content": "用一句话总结"})
resp2 = client.chat.completions.create(model="deepseek-chat", messages=messages)
print(resp2.choices[0].message.content)
```

### 2.2 流式输出

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一首五言绝句"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### 2.3 使用 deepseek-reasoner（R1）

```python
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "9.9 和 9.11 哪个大？请逐步推理。"}],
)
msg = resp.choices[0].message
# 部分 SDK 版本字段名可能为 reasoning_content，以官方文档为准
reasoning = getattr(msg, "reasoning_content", None)
answer = msg.content
if reasoning:
    print("【思考】\n", reasoning[:500], "...\n")
print("【答案】\n", answer)
```

---

## 3. 封装成可复用模块

建议项目结构：

```text
my_ai_app/
  llm_client.py      # 统一封装
  prompts/           # 系统提示模板
  app.py             # 业务入口
  .env               # 密钥（勿提交 git）
```

**llm_client.py** 示例：

```python
import os
from openai import OpenAI

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )
    return _client

def chat(user: str, system: str = "", history: list | None = None, **kwargs) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    resp = get_client().chat.completions.create(
        model=kwargs.pop("model", os.getenv("LLM_MODEL", "deepseek-chat")),
        messages=messages,
        temperature=kwargs.pop("temperature", 0.3),
        max_tokens=kwargs.pop("max_tokens", 2048),
        **kwargs,
    )
    return resp.choices[0].message.content
```

**.env**（配合 `python-dotenv`）：

```env
DEEPSEEK_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
# 本地 Ollama 时：
# LLM_BASE_URL=http://localhost:11434/v1
```

---

## 4. Prompt 工程基础（够用的 5 条）

1. **角色 + 任务 + 格式**：「你是 SQL 专家。根据表结构生成 SELECT。只输出 SQL，不要解释。」
2. **Few-shot**：在 prompt 里给 1～2 个输入输出示例。
3. **分步**：复杂任务写「步骤 1…步骤 2…」。
4. **约束**：长度、语言、禁止项（「不要编造链接」）。
5. **R1/推理模型**：直接问难题即可，不必写「请一步步思考」——模型会自动 CoT；**展示层**可隐藏思考过程。

**JSON 输出**（便于程序解析）：

```python
system = """只输出合法 JSON，不要 markdown 代码块。
格式：{"title": str, "tags": [str]}"""
```

若模型仍输出 markdown，后处理去掉 ` ```json ` 包裹。

---

## 5. 结构化输出与 Function Calling

DeepSeek API 支持 **工具调用（Tools）**，类似 OpenAI function calling：模型返回 `tool_calls`，你的程序执行函数后再把结果塞回 `messages`。

典型流程：

```text
用户问天气 → 模型返回 call get_weather(city) → 你执行 API → 把结果 role=tool 传回 → 模型生成自然语言答复
```

入门可先手写 **JSON schema prompt**；业务稳定后再上 Tools。详见 [DeepSeek API 文档](https://api-docs.deepseek.com/)。

---

## 6. RAG 最小闭环（检索增强生成）

**RAG（Retrieval-Augmented Generation）**  
先从你的文档库检索相关段落，再塞进 prompt，减少「胡说」。

```text
用户问题 → 向量检索 Top-K 段落 → 拼进 system/user → LLM 生成
```

**最小依赖**：

```bash
pip install openai chromadb sentence-transformers
```

**流程伪代码**：

```python
# 1. 离线：文档切块 → embedding → 存入 Chroma
# 2. 在线：
query = "报销流程是什么？"
docs = collection.query(query_texts=[query], n_results=3)
context = "\n".join(docs["documents"][0])
prompt = f"依据以下资料回答，不知道就说不知道：\n{context}\n\n问题：{query}"
answer = chat(prompt, system="你是公司内部助手")
```

**入门建议**：先用 **100 页以内** PDF/Markdown 验证；embedding 可用 `BAAI/bge-small-zh-v1.5`。

---

## 7. 错误处理与生产要点

| 现象 | 处理 |
|------|------|
| 429 Rate Limit | 指数退避重试；降并发 |
| 超时 | `timeout=60`；流式更易感知卡住 |
| 内容过滤 | 换表述；检查是否触发安全策略 |
| 幻觉 | RAG + 「仅根据上文」+ 降低 temperature |
| 成本飙升 | 缩短 system；缓存常见问题；小模型做路由 |

**重试示例**：

```python
import time
from openai import RateLimitError

def chat_with_retry(**kwargs):
    for i in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            time.sleep(2 ** i)
    raise RuntimeError("rate limit exceeded")
```

---

## 8. 小项目练手（按顺序做）

| # | 项目 | 练到什么 |
|---|------|----------|
| 1 | 命令行问答 CLI | messages、环境变量 |
| 2 | 流式 Web 聊天（Flask/FastAPI + SSE） | streaming |
| 3 | 带历史的会话（JSON 文件存 messages） | 多轮 |
| 4 | PDF 问答（RAG） | embedding + 检索 |
| 5 | SQL 生成器（给 schema + 自然语言） | prompt + 解析 |

---

## 9. 与「训练」的衔接

| 你现在做的 | 之后训练用的 |
|------------|--------------|
| 多轮 messages 格式 | SFT 数据集格式 |
| system prompt 设计 | 指令模板设计 |
| 不好的回答案例 | 改进微调数据 |
| RAG 检索不到 | 可能需要微调领域模型 |

下一步：[guide-02 微调入门](/part-06-practice/02-finetuning)

---

## 10. 自检

- [ ] 写过带 system prompt 的多轮对话
- [ ] 跑通过 stream 模式
- [ ] 试过 `deepseek-reasoner` 并理解思考/答案分离
- [ ] 封装了可复用的 `chat()` 函数
- [ ] （可选）完成一个最小 RAG demo

---

## 示例文件

- [`examples/part-06-practice/01-llm-client/llm_client.py`](/examples/part-06-practice/01-llm-client/llm_client.py)
- [`examples/part-06-practice/01-llm-client/.env.example`](/examples/part-06-practice/01-llm-client/.env.example)

运行：`python examples/part-06-practice/01-llm-client/llm_client.py`
