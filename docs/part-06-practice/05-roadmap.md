# AI 训练与开发：从 DeepSeek 部署到进阶的完整路线图

> **本文**：把 **实战指南** 与 **原理文档** 串成一条学习路径，按周推进即可快速入门。  
> **环境**：Windows 为主，训练建议 WSL2 / 云 GPU。

---

## 1. 文档地图

### 1.1 实战入门（先做）

| 顺序 | 文档 | 产出 |
|------|------|------|
| 0 | [guide-00 部署](/part-06-practice/00-deployment) | API 或本地模型跑通 |
| 1 | [guide-01 应用开发](/part-06-practice/01-inference) | Python 封装 + 可选 RAG |
| 2 | [guide-02 微调 SFT/LoRA](/part-06-practice/02-finetuning) | 第一个 LoRA 适配器 |
| 3 | [guide-03 数据与评测](/part-06-practice/03-data-evaluation) | Golden Set + 评测脚本 |
| 4 | [guide-04 RL 进阶](/part-06-practice/04-rl-roadmap) | 理解 GRPO / 可选实验 |

### 1.2 原理深读（理解「为什么」）

| 文档 | 何时读 |
|------|--------|
| [deepseek-training-guide-zh.md](/part-07-theory/training-guide) | 第 1～2 周后，建立全局 |
| [deepseek-v3-pretrain-posttrain-deep-dive-zh.md](/part-07-theory/v3-pretrain-posttrain) | 好奇预训练/成本时 |
| [deepseek-v3-architecture-deep-dive-zh.md](/part-07-theory/v3-architecture) | 做部署优化/选型时 |
| [deepseek-grpo-rl-deep-dive-zh.md](/part-07-theory/grpo-rl) | 做 RL 实验前 |
| [deepseek-r1-pipeline-deep-dive-zh.md](/part-07-theory/r1-pipeline) | 设计多阶段训练时 |
| [deepseek-distill-reproduction-deep-dive-zh.md](/part-07-theory/distill-reproduction) | 复现 Distill 时 |

---

## 2. 能力阶梯（你在哪一级）

```text
L0  会用 API 聊天
L1  能写应用（多轮、流式、RAG）
L2  能 LoRA 微调 + 评测
L3  能造数据（Teacher + 验证 + 版本化）
L4  能跑 GRPO / 玩具 RL
L5  理解 DeepSeek 全流水线（读论文级文档）
L6  机构级预训练 / 大规模 RL（非个人目标）
L7  能构建迷你 LLM（Part 8：分词→预训练→SFT）
L8  能开发 Agent（Part 9：ReAct→RAG→多 Agent）
```

**快速入门目标**：**8 周内达到 L2～L3**；L4～L5 按兴趣延伸；**L7～L8** 对应「造模型」与「做 Agent」。

---

## 3. 八周学习计划（可压缩为 4 周全职）

### 第 1 周：部署 + 第一次调用

| 天 | 任务 | 文档 |
|----|------|------|
| 1～2 | 注册 DeepSeek API，跑 smoke_test | guide-00 §2 |
| 3 | 装 Ollama 或 LM Studio，本地 7B | guide-00 §3～4 |
| 4～5 | Python 封装 `chat()`，多轮 + 流式 | guide-01 §2～3 |
| 6～7 | 做小 CLI 或 Streamlit 聊天页 | guide-01 §8 项目 1～2 |

**里程碑**：L0 → L1 前半。

---

### 第 2 周：Prompt + 小应用

| 天 | 任务 |
|----|------|
| 1～2 | 系统 prompt 模板；试 `deepseek-reasoner` |
| 3～4 | 最小 RAG（50 页 PDF + Chroma） |
| 5～7 | 完成一个 **垂直小应用**（如 FAQ 机器人） |

**里程碑**：稳定 L1。

---

### 第 3 周：数据 + 第一次微调

| 天 | 任务 | 文档 |
|----|------|------|
| 1～2 | 写数据规格；手工 100 条 Alpaca | guide-03 §4 |
| 3 | 划分 train/val/test | guide-03 §4 Step 6 |
| 4～5 | LLaMA-Factory WebUI LoRA 7B | guide-02 §4 |
| 6～7 | 对比微调前后 20 题 | guide-02 §8 |

**里程碑**：L2。

---

### 第 4 周：Teacher 蒸馏 + 评测

| 天 | 任务 |
|----|------|
| 1～3 | API 批量生成 500～2000 条（带过滤） |
| 4 | 建 100 条 Golden Set + 自动评测脚本 |
| 5～7 | 第二轮 LoRA；记录实验表 |

**里程碑**：L3 入门。

---

### 第 5～6 周：原理 + 优化（并行阅读）

| 任务 |
|------|
| 读 [training-guide](/part-07-theory/training-guide) 一遍 |
| 读 [distill 复现](/part-07-theory/distill-reproduction) §7～8 |
| 优化 RAG / 应用；尝试 vLLM（WSL2） |

**里程碑**：L3 巩固 + L5 理论骨架。

---

### 第 7～8 周：进阶（选修）

| 选项 A | 选项 B |
|--------|--------|
| TRL GRPO + GSM8K 子集 | TinyZero 2 卡实验 |
| 读 [GRPO 深挖](/part-07-theory/grpo-rl) | 读 [R1 pipeline](/part-07-theory/r1-pipeline) |

**里程碑**：L4 体验；不必刷榜。

---

## 4. 工具链推荐（一套够用）

| 环节 | 工具 |
|------|------|
| 推理 API | DeepSeek 官方 / Ollama |
| 应用 | Python 3.11 + OpenAI SDK + FastAPI |
| RAG | Chroma + bge-small-zh |
| 微调 | LLaMA-Factory（主）+ Unsloth（快试） |
| 评测 | 自建 Golden + lm-eval-harness 子集 |
| 实验跟踪 | wandb 或本地 CSV |
| RL 试验 | TRL GRPO / TinyZero |
| 云 GPU | AutoDL 等按小时租 4090/A100 |

---

## 5. 项目.portfolio 建议（求职/作品集）

按难度递进，**每个都要可 demo + README**：

1. **DeepSeek Chat CLI** — 多轮 + 流式  
2. **企业 PDF 问答** — RAG + DeepSeek API  
3. **领域 LoRA 助手** — 自造 1k 数据 + LLaMA-Factory + 前后对比报告  
4. **（加分）数学蒸馏 mini** — Teacher 生成 + sympy 过滤 + 7B LoRA + GSM8K 子集分数  

---

## 6. 常见路径误区

| 误区 | 正解 |
|------|------|
| 一上来学 Transformer 公式 | 先 L0～L2 跑通再补理论 |
| 没评测狂加数据 | 先 100 条 Golden Set |
| 本地硬跑 671B | API + Distill 7B |
| 跳过 SFT 直接 RL | 先 SFT 再 GRPO |
| 只看 loss 不看样例 | 每轮固定 20 题人工对比 |

---

## 7. 环境与账号 checklist

```text
[ ] DeepSeek API Key + 余额
[ ] Python 3.10+ venv
[ ] Git
[ ] （可选）Ollama / LM Studio
[ ] （微调）WSL2 Ubuntu + CUDA 或云 GPU 账号
[ ] HuggingFace 账号（下权重）
[ ] wandb 账号（可选）
```

**Windows 微调路径**：  
- 轻量：Unsloth 4bit + 7B  
- 正规：WSL2 装 LLaMA-Factory + CUDA  
- 或：云 GPU 租机，本地只写代码

---

## 8. 每日 30 分钟极简习惯

1. **5 min** — 读一条术语或论文摘要  
2. **15 min** — 改 prompt / 标 5 条数据 / 看 10 条生成样例  
3. **10 min** — 记实验笔记（今天改了什么、分数变多少）

---

## 9. 下一步行动（今天就做）

1. 打开 [guide-00](/part-06-practice/00-deployment) §2 或 §3，完成 **smoke test**  
2. _fork 本仓库 `docs/deepseek/` 到本地书签_  
3. 在 `D:\code\code_base` 或单独目录建 `my_ai_lab/`，放入 `llm_client.py`  

完成第 1 周后，回到本文 **第 2 周表格** 继续。

---

## 10. 获取帮助

| 问题类型 | 去向 |
|----------|------|
| API 报错 | [DeepSeek API 文档](https://api-docs.deepseek.com/) |
| LLaMA-Factory | [GitHub Issues](https://github.com/hiyouga/LLaMA-Factory) |
| 原理细节 | 本目录 `deepseek-*-deep-dive-zh.md` |
| 社区 | HuggingFace Discord、知乎 / Reddit LocalLLaMA |

祝学习顺利：**先跑起来，再变强。**
