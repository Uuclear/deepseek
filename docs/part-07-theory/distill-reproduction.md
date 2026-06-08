# DeepSeek-R1 蒸馏系列与复现路径：面向小白的深度解读

> 本文基于 DeepSeek-R1 论文（arXiv:2501.12948）Distillation 章节、官方 GitHub README（DeepSeek-R1 / DeepSeek-V3）、以及社区复现综述（*100 Days After DeepSeek-R1*，arXiv:2505.00551）整理。  
> 目标：零基础也能读懂「R1 蒸馏是什么、官方怎么做的、社区怎么复现、个人能做什么」。  
> 生成日期：2026-06-08

---

## 阅读导航

| 章节 | 内容 |
|------|------|
| [一、术语词典](#一术语词典先读这里) | 所有缩写与概念解释 |
| [二、蒸馏原理](#二蒸馏原理teacher-student-为何-r1小模型-更好) | Teacher-Student、为何蒸馏优于小模型直接 RL |
| [三、Distill 系列型号表](#三distill-系列完整型号表) | 6 个型号、基座、许可、Benchmark |
| [四、蒸馏数据如何生成](#四蒸馏数据如何生成) | Prompt 来源、CoT 格式、过滤规则 |
| [五、开源权重与 API](#五开源权重与-api-使用) | HuggingFace、DeepSeek 平台 |
| [六、消费级 GPU 部署](#六消费级-gpu-推理部署要点) | 量化、vLLM、上下文长度 |
| [七、社区复现对比](#七社区复现项目对比) | Open-R1、TinyZero、SimpleRL 等与官方差异 |
| [八、个人能复现什么](#八个人小团队能复现什么不能复现什么) | Realistic 清单 |
| [九、法律与许可](#九法律与许可) | MIT、Apache、Llama 许可对商用/蒸馏的含义 |

---

## 一、术语词典（先读这里）

### 1.1 模型与训练

**LLM（Large Language Model，大语言模型）**  
能理解和生成文本的神经网络。ChatGPT、DeepSeek、Claude 都属于 LLM。

**Teacher Model（教师模型）**  
能力更强、用来「出题并写标准解题过程」的大模型。在 R1 蒸馏里，教师就是 **DeepSeek-R1**（671B 总参数、每次激活约 37B 的 MoE 模型）。

**Student Model（学生模型）**  
被训练的小模型，目标是模仿教师的推理方式。R1-Distill 系列的学生是 Qwen / Llama 等 **Dense（稠密）** 小模型。

**Knowledge Distillation（知识蒸馏）**  
把大模型的能力「压缩」到小模型：不是只学最终答案，而是学 **完整的思考过程（Chain-of-Thought，CoT）**。

**SFT（Supervised Fine-Tuning，监督微调）**  
用「输入 → 标准输出」配对数据继续训练模型。R1 蒸馏 **只做 SFT，不做 RL**。

**RL（Reinforcement Learning，强化学习）**  
模型自己生成答案，根据对错获得奖励/惩罚，反复优化。DeepSeek-R1 本体靠大规模 RL 获得推理能力；蒸馏小模型 **官方未做这一步**（但论文指出加上 RL 还能再涨分）。

**RLVR（Reinforcement Learning from Verifiable Rewards）**  
奖励来自 **可自动验证** 的任务（数学题对错、代码能否跑通），不需要人工打分。DeepSeek-R1-Zero 就是典型 RLVR。

**GRPO（Group Relative Policy Optimization）**  
DeepSeek 在 R1 训练中使用的 RL 算法，比传统 PPO 更省算力。社区复现 R1-Zero 时常用 GRPO 或其变体。

**CoT（Chain-of-Thought，思维链）**  
模型在给出最终答案前，逐步写出推理过程。R1 系列用 `...` 标签包裹思考内容。

**Rejection Sampling（拒绝采样 / 筛选采样）**  
对同一道题让模型生成多个回答，**只保留答对的**，丢弃错的——像「多次模考，只收录满分卷」。

**Pass@1**  
每题只采样 1 次，看是否答对的比例。论文蒸馏评测用 temperature=0.6、top-p=0.95 采样。

**cons@64**  
每题采样 64 次，取 **多数投票（majority vote）** 后的准确率。AIME 等难题上 cons@64 通常高于 pass@1。

**MoE（Mixture of Experts，混合专家）**  
大模型内部有很多「专家子网络」，每次只激活一部分，总参数大但推理成本相对可控。DeepSeek-R1 / V3 是 MoE；Distill 系列是 Dense。

**Dense Model（稠密模型）**  
所有参数每次推理都会用到。7B、14B、32B 的 Qwen/Llama 都是 Dense，更适合个人 GPU 部署。

### 1.2 评测 Benchmark

| 名称 | 测什么 |
|------|--------|
| **AIME 2024** | 美国数学邀请赛，高难度奥数级 |
| **MATH-500** | 500 道数学题的子集 |
| **GPQA Diamond** | 研究生级别科学问答 |
| **LiveCodeBench** | 真实编程竞赛题（论文用 2024-08 至 2025-01 数据） |
| **Codeforces rating** | 在 Codeforces 平台风格的编程能力评分 |

### 1.3 部署相关

**量化（Quantization）**  
把模型权重从 FP16/BF16 压到 INT8、INT4 等，显存减半或更多，速度更快，精度略降。

**vLLM**  
高性能 LLM 推理框架，支持张量并行（多卡切模型）、连续批处理。

**Tensor Parallel（张量并行，TP）**  
把一个大模型切到多张 GPU 上同时算。32B 模型常用 `--tensor-parallel-size 2`（2 卡）。

**Context Length（上下文长度）**  
模型一次能读+写的最大 token 数。R1 系列评测上限 **32,768 tokens**；Distill 模型继承基座上下文（通常 32K 或 128K，以 HuggingFace 卡片为准）。

---

## 二、蒸馏原理：Teacher-Student，为何 R1→小模型 比小模型直接 RL 更好

### 2.1 用一句话说清

**DeepSeek-R1 蒸馏 = 用 R1 老师生成的 80 万条「题目 + 完整思考过程 + 答案」，对 Qwen/Llama 小学生做 SFT，让小模型学会像 R1 一样一步步推理。**

官方 **刻意不做 RL**，目的是证明：光蒸馏就已经很强；RL 留给社区探索。

### 2.2 Teacher-Student 流程（官方）

```
┌─────────────────────────────────────────────────────────────┐
│  Teacher: DeepSeek-R1（671B MoE，经 RL+SFT 训练）            │
│  输入：各类 prompt（数学/代码/写作/翻译…）                    │
│  输出：推理过程 + 最终答案                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ Rejection Sampling + 过滤
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  蒸馏数据集：约 800k 样本                                     │
│  · 600k 推理类（数学/代码/STEM…）                             │
│  · 200k 非推理类（写作/QA/翻译…，简单题可无 CoT）             │
└──────────────────────────┬──────────────────────────────────┘
                           │ 仅 SFT（无 RL）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Student: Qwen2.5 / Llama 小模型（1.5B～70B）                │
│  → DeepSeek-R1-Distill-* 系列                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 为何「R1 蒸馏到小模型」优于「小模型自己跑大规模 RL」

论文 **§4.1 Distillation v.s. Reinforcement Learning** 做了直接对比实验（基座：**Qwen-32B-Base**）：

| 模型 | 训练方式 | AIME 2024 pass@1 | AIME cons@64 | MATH-500 | GPQA Diamond | LiveCodeBench |
|------|----------|------------------|--------------|----------|--------------|---------------|
| QwQ-32B-Preview | Qwen 官方推理预览版 | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| **DeepSeek-R1-Zero-Qwen-32B** | 小模型 **直接 RL**（>10K steps，数学+代码+STEM） | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| **DeepSeek-R1-Distill-Qwen-32B** | **R1 蒸馏 + 仅 SFT** | **72.6** | **83.3** | **94.3** | **62.1** | **57.2** |

**关键数字解读：**

- 32B 小模型自己 RL **10,000+ 步**，AIME pass@1 只有 **47.0%**，和 QwQ-32B-Preview（50.0%）同一量级——**算力砸下去，够不到蒸馏**。
- 同样 32B 基座，R1 蒸馏 pass@1 **72.6%**，AIME 上 **+25.6 个百分点**，LiveCodeBench **+17.0**——**蒸馏全面碾压小模型 RL**。
- 7B 蒸馏 AIME **55.5%**，已超过 GPT-4o（9.3%）和 Claude-3.5-Sonnet（16.0%）在该榜上的 pass@1。
- 14B 蒸馏 **69.7%**，**全面超过** QwQ-32B-Preview（50.0%）——**更小模型 + 更好老师 > 更大模型自己摸索**。

**论文结论（两句话）：**

1. **蒸馏强模型到小模型，性价比高**；小模型依赖论文同款大规模 RL，**算力需求巨大且可能仍不如蒸馏**。
2. **要突破智能上限**，仍需要更强的基座 + 更大规模 RL——蒸馏是「把已有天花板教给小学生」，不是「造新天花板」。

### 2.4 为什么Teacher 的「推理模式」更值钱？

R1 在 **671B 级 MoE + 海量 RL** 上涌现了：自我验证、反思、长 CoT、分步拆解等模式。这些模式作为 **监督信号** 写入小模型，比让小模型在有限 RL 预算里 **从零探索** 效率高得多——类似「奥数金牌手写详解」vs「普通学生自己刷题摸索」。

论文还提到：对蒸馏模型 **再加 RL 还能显著提升**（未公开具体数字），说明蒸馏是下限很高的起点。

---

## 三、Distill 系列完整型号表

官方 2025-01-20 发布 **6 个** 开源蒸馏 checkpoint（[GitHub README](https://github.com/deepseek-ai/DeepSeek-R1)）。

### 3.1 型号、基座、许可、下载

| 官方名称 | 参数量 | 基座模型 | HuggingFace | 继承许可 |
|----------|--------|----------|-------------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | Apache 2.0（Qwen） |
| DeepSeek-R1-Distill-Qwen-7B | 7B | [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Apache 2.0 |
| DeepSeek-R1-Distill-Llama-8B | 8B | [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | Llama 3.1 Community License |
| DeepSeek-R1-Distill-Qwen-14B | 14B | [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | Apache 2.0 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | Apache 2.0 |
| DeepSeek-R1-Distill-Llama-70B | 70B | [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [链接](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | Llama 3.3 Community License |

**注意：**

- 官方 **微调了 config 和 tokenizer**，README 明确要求：**请用 DeepSeek 发布的权重与配置**，不要直接当普通 Qwen/Llama 用。
- Llama-70B 基座选 **3.3-Instruct** 而非 3.1，因 3.3 推理能力略强。
- Qwen 小尺寸用 **Math** 系列基座（数学预训练增强），14B/32B 用通用 Qwen2.5。

### 3.2 Benchmark 完整表（论文 Table 5 / README）

评测设置：max gen **32,768 tokens**；temperature **0.6**，top-p **0.95**；pass@1 为多次采样平均。

| 模型 | AIME 2024 pass@1 | AIME cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | Codeforces rating |
|------|------------------|--------------|-----------------|---------------------|----------------------|-------------------|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| OpenAI o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| QwQ-32B-Preview | 44.0 | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| **R1-Distill-Qwen-1.5B** | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| **R1-Distill-Qwen-7B** | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| **R1-Distill-Qwen-14B** | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| **R1-Distill-Qwen-32B** | **72.6** | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| **R1-Distill-Llama-8B** | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| **R1-Distill-Llama-70B** | 70.0 | **86.7** | **94.5** | **65.2** | **57.5** | 1633 |

**怎么选型号（实用建议）：**

| 你的场景 | 推荐 |
|----------|------|
| 单卡 8～12GB，本地玩推理 | 1.5B（INT4）或 7B（INT4/INT8） |
| 单卡 24GB | 7B FP16 或 14B INT4 |
| 双卡 24GB×2 | 32B INT4 + TP=2 |
| 要接近 o1-mini、有 2×80GB | 32B FP16 或 70B INT4 |
| 商用且不想碰 Llama 许可 | 优先 Qwen 系（Apache 2.0） |

---

## 四、蒸馏数据如何生成

> 官方 **80 万条蒸馏数据未公开下载**；以下来自论文 §2.3.3、§2.4 及社区综述对方法的还原。**社区复现用的是「类似流程 + 自建数据」**，不是同一份数据。

### 4.1 数据从哪来（Prompt 来源）

R1 本体的 SFT 数据分两块，蒸馏 **复用同一套约 800k**：

| 类别 | 规模 | Prompt 来源 | 生成方式 |
|------|------|-------------|----------|
| **推理类** | ~**600k** | 数学、代码、STEM、逻辑等；RL 收敛后的 checkpoint 做 **Rejection Sampling** | 每题多次采样，**只留正确答案**；部分用 **生成式奖励模型**（把标准答案与模型预测喂给 DeepSeek-V3 判断） |
| **非推理类** | ~**200k** | 写作、事实 QA、自我认知、翻译等；**复用 DeepSeek-V3 的部分 SFT 数据** | 较复杂问题先让 V3 生成 CoT 再答；简单问候类 **不含 CoT** |

推理类 prompt 在 RL 阶段还扩展了原先「只能规则判分」以外的数据；非推理类保证蒸馏模型 **不只会做题，还会聊天**。

### 4.2 CoT 格式

DeepSeek-R1 系列的标准格式：

```text

（此处是模型的逐步推理：分解问题、验算、反思、自我纠正…）

（此处是面向用户的最终答案，数学常含 \boxed{}）
```

**使用建议（官方 README）：**

1. **不要加 system prompt**；指令全放在 user prompt。
2. 数学题建议加：「Please reason step by step, and put your final answer within \boxed{}。」
3. temperature **0.5～0.7**（推荐 **0.6**），避免死循环或胡言乱语。
4. 若模型跳过思考，可在输出开头 **强制** 以 `\n` 起头。
5. 评测时 **多次采样取平均**（pass@1），不要只用 greedy decoding（长 CoT 模型易重复）。

Distill 模型学的就是这种「带 thinking 块的长输出」分布；上下文需预留 **足够长**（评测 32K）。

### 4.3 过滤规则（Rejection Sampling + 质量过滤）

论文明确提到的过滤：

| 规则 | 目的 |
|------|------|
| 每 prompt **多次采样，仅保留正确回答** | 保证监督信号质量（Rejection Sampling 核心） |
| 去掉 **语言混杂** 的 CoT | R1-Zero 期常见问题，影响可读性 |
| 去掉 **过长段落** | 控制训练稳定性与可读性 |
| 去掉含 **大段 code block** 的混乱输出 | 避免格式污染 |
| 推理题：规则验证 + 必要时 **V3 作 judge** | 数学/代码可自动判分；开放题用强模型评判 |
| 非推理简单 query：**不提供 CoT** | 避免「你好」也长篇思考 |

社区数据集（Open-R1、OpenThoughts 等）在此基础上还常做：**去重（embedding / n-gram）**、**Math Verify 验算**、**代码单元测试**、**LLM-as-judge**、**难度采样（去掉过易/过难）** 等——见 [七、社区复现](#七社区复现项目对比)。

### 4.4 训练细节（蒸馏阶段）

- **方法**：**仅 SFT**，**无 RL**（论文 §2.4 原文）。
- **数据量**：约 **800k**（600k + 200k）。
- **基座**：见上表；Llama 选 3.3-70B-Instruct。
- **与 R1 本体区别**：R1 本体还有 cold-start → RL → 再 SFT → 再 RL 的多阶段；蒸馏是 **一步 SFT 到位**。

---

## 五、开源权重与 API 使用

### 5.1 HuggingFace 权重

所有 Distill 模型：`deepseek-ai/DeepSeek-R1-Distill-*`（见第三节链接）。

**加载示例（Transformers）：**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

**注意：** 完整 **DeepSeek-R1（671B）** 在 README 中注明 *Transformers 尚未直接支持*，需 SGLang / vLLM 等；**Distill 系列与普通 Qwen/Llama 一样用法**。

### 5.2 官方 API（Teacher R1，非 Distill 权重）

| 入口 | 用途 |
|------|------|
| [chat.deepseek.com](https://chat.deepseek.com) | 网页聊天，开启 **DeepThink** 即用 R1 推理 |
| [platform.deepseek.com](https://platform.deepseek.com) | **OpenAI 兼容 API**，可调用 DeepSeek-R1 |

API 适合：**自己蒸馏时调用 R1 生成 CoT**（社区 Open-R1、OpenThoughts 等的数据管线即如此），无需本地 671B。

**自建蒸馏的数据流：**

```
你的 prompt 数据集 → 调用 DeepSeek-R1 API 生成 CoT → 过滤/验真 → SFT 训练小模型
```

成本：按 token 计费；Open-R1 的 220k 数学集、Mixture-of-Thoughts 350k 等是公开参考规模。

### 5.3 相关开源 Teacher / 数据（非官方 800k）

| 资源 | 规模 | 说明 |
|------|------|------|
| [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | 350k | Open-R1 Step 1，R1 蒸馏 trace，含数学/代码/科学 |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 220k | 数学题 + R1 CoT，Math Verify 过滤 |
| [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | 114k | 多领域 R1 合成 |
| [GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO) | 817 | 极简高质量子集，证明「少而精」也有效 |

---

## 六、消费级 GPU 推理部署要点

### 6.1 显存粗算（仅供选型）

公式直觉：**参数量 × 每参数字节数 ≈ 权重显存**（还需加 KV Cache、激活值）。

| 精度 | 每参数字节 | 7B 权重约 | 32B 权重约 |
|------|------------|-----------|------------|
| FP16/BF16 | 2 bytes | ~14 GB | ~64 GB |
| INT8 | 1 byte | ~7 GB | ~32 GB |
| INT4 (AWQ/GPTQ) | ~0.5 byte | ~4 GB | ~16 GB |

**KV Cache** 随 **上下文长度 × batch** 增长；R1 类模型 CoT 很长，**实际显存远高于「只载权重」**。

### 6.2 各型号部署参考

| 模型 | 推荐方式 | 典型硬件 |
|------|----------|----------|
| 1.5B | llama.cpp / Ollama INT4；Transformers | 4～8 GB 显存或 CPU |
| 7B | vLLM / Ollama；INT4 单卡 | 8～16 GB |
| 8B Llama | 同 7B | 8～16 GB |
| 14B | INT4 单卡或 FP16 24GB | 16～24 GB |
| 32B | **vLLM TP=2** + INT4，或 INT4 单卡 24GB 勉强 | 24～48 GB（推荐 2×24GB） |
| 70B | INT4 + TP=2～4 | 48 GB+（通常 2×A100 80GB 或 4×24GB） |

### 6.3 vLLM 官方示例（32B）

```shell
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enforce-eager
```

**参数说明：**

- `--tensor-parallel-size 2`：2 卡张量并行。
- `--max-model-len 32768`：与论文评测一致；**显存不够就降低**（如 8192、16384），但长 CoT 可能被截断。
- `--enforce-eager`：部分模型/驱动下避免 CUDA graph 问题；生产可试去掉以提速。

**SGLang 替代：**

```bash
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --trust-remote-code --tp 2
```

### 6.4 量化工具

| 工具 | 说明 |
|------|------|
| **AWQ / GPTQ** | 常用 INT4 权重；HuggingFace 上常有社区量化版 |
| **llama.cpp / Ollama** | 本地最省心；7B/8B 友好 |
| **bitsandbytes INT8/INT4** | Transformers 里 `load_in_4bit=True` |

量化对 **数学推理** 可能有 1～3 点 pass@1 损失，需在你任务上实测。

### 6.5 推理调参 checklist

- [ ] temperature = **0.6**，top_p = **0.95**
- [ ] 不加 system prompt
- [ ] 数学题加 step-by-step + `\boxed{}` 提示
- [ ] 必要时强制 `\n` 开头
- [ ] 长推理预留 **16K～32K** context
- [ ] 生产环境限制 **max_tokens**，防止 CoT 无限拉长

---

## 七、社区复现项目对比

> 官方 **未公开** R1 完整训练代码与 800k 数据；社区目标是 **复现能力**，不是 bitwise 复刻。  
> 综述来源：*100 Days After DeepSeek-R1*（arXiv:2505.00551）及各自 GitHub / 技术报告。

### 7.1 总览对照表

| 项目 | 主攻方向 | 基座示例 | 核心方法 | 与官方差异 |
|------|----------|----------|----------|------------|
| **官方 R1-Distill** | 蒸馏小模型 | Qwen/Llama 1.5B～70B | R1 生成 **800k** → **仅 SFT** | 数据不公开；Benchmark 最高 |
| **[Open-R1](https://github.com/huggingface/open-r1)** | 全链路复现（蒸馏→R1-Zero→多阶段） | Qwen2.5-Math-7B 等 | 自建 **Mixture-of-Thoughts 350k**、OpenR1-Math-220k；Step1 已复现 7B 蒸馏 | 数据公开；规模小于 800k；OpenR1-Distill-7B ≈ 官方 7B 但未完全一致 |
| **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)** | **R1-Zero 行为**（非完整 R1） | Qwen2.5-**3B**-Base | 倒计时/乘法 **玩具任务** + **PPO**（veRL） | **不是**数学 AIME 级复现；证明小模型 **RL 涌现** self-verify；**<$30**；仓库已归档，推荐用 veRL |
| **[SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason)** (HKUST) | 简化 **RLVR** | Qwen2.5-Math-7B | **8k** 数学题 + **PPO**（非 GRPO） | 无 cold-start、无多阶段；AIME ~36.7；**远低**官方蒸馏 7B（55.5） |
| **[DeepScaleR](https://github.com/agentica-project/deepscaler)** | 小模型 **RL Scaling** | R1-Distill-Qwen-1.5B 等 | **40k** 竞赛数学 + **GRPO** | 在 **蒸馏模型上再 RL**；1.5B AIME ~43.1；验证「蒸馏+RL」路线 |
| **[OpenThoughts](https://open-thoughts.ai)** | 公开蒸馏数据 | Qwen 7B/32B Instruct | **114k** R1 CoT 多领域 | 数据质量高但规模小；32B SFT AIME ~68.0 vs 官方 72.6 |
| **[LIMO](https://huggingface.co/datasets/GAIR/LIMO)** | **极少数据** SFT | Qwen2.5-32B-Instruct | 仅 **817** 条精选 | AIME ~57.1；证明 curation > scale，但难复现官方上限 |
| **[Light-R1](https://github.com/...) ** | 蒸馏 + 二阶段 RL | 自研 14B 流程 | 76k SFT + GRPO | 接近官方「蒸馏后再 RL」思路 |
| **[Skywork-OR1](https://github.com/...) ** | 大模型 RLVR | R1-Distill-Qwen-32B 起点 | 105k + GRPO | 从 **蒸馏模型** 继续 RL，非从零 R1 |

### 7.2 三条路线对比（心智模型）

```
路线 A【官方蒸馏】  R1 Teacher → 800k CoT → SFT → Distill 模型（无 RL）
路线 B【社区蒸馏】  R1 API → 公开数据集(114k~350k) → SFT → 接近但通常略低于官方
路线 C【社区 RL-Zero】 小基座 → 可验证奖励 RL → 涌现推理，但 AIME 远低于路线 A
路线 D【蒸馏 + RL】  官方或社区 Distill → 再加 GRPO/PPO → 论文称还有提升空间（DeepScaleR 等）
```

### 7.3 Open-R1 三阶段（与官方 pipeline 映射）

| Open-R1 步骤 | 对应官方 | 状态（截至社区 blog） |
|--------------|----------|------------------------|
| Step 1：复现 R1-Distill | §2.4 蒸馏 | ✅ Mixture-of-Thoughts + OpenR1-Distill-7B |
| Step 2：复现 R1-Zero 纯 RL | §2.2 R1-Zero | 🔄 进行中，需自建大规模 math/code 数据 |
| Step 3：Base → SFT → RL 全流程 | §2.3 完整 R1 | 🔄 进行中 |

**与官方差异：** Open-R1 **完全开源数据与脚本**；官方 **800k 未发布**；Open-R1 用 **GRPO 生态（TRL 等）** 而非 DeepSeek 内部框架。

### 7.4 TinyZero vs 官方 R1-Zero

| 维度 | 官方 R1-Zero | TinyZero |
|------|--------------|----------|
| 基座 | DeepSeek-V3-Base **671B** | Qwen2.5-**3B** |
| 任务 | 数学、代码、STEM 真实题 | Countdown、乘法 **合成小游戏** |
| 算法 | **GRPO** | **PPO**（veRL） |
| 算力 | 集群级 | **2 卡 GPU，~<$30** |
| 目标 | SOTA 推理 | 教育性：**Aha moment**（自验证、搜索） |
| AIME | 79.8（完整 R1） | **不评测 AIME** |

TinyZero **不能**替代官方 R1 或 Distill；它证明 **RL 涌现** 可发生在极小任务+小模型上。

### 7.5 SimpleRL vs 官方

| 维度 | 官方 Distill-7B | SimpleRL-Zero-7B |
|------|-----------------|------------------|
| 训练 | 800k SFT | **8k** RL（PPO） |
| AIME 2024 | **55.5** | ~**36.7** |
| MATH-500 | **92.8** | ~**77.4** |
| 结论 | 蒸馏主导 | 简化 RL **有效但远弱于蒸馏** |

SimpleRL 价值：**低资源验证 RLVR pipeline**，不是刷榜方案。

### 7.6 社区蒸馏数据集规模对比

| 项目 | SFT 规模 | 数学 | 代码 | 其他推理 | 非推理 |
|------|----------|------|------|----------|--------|
| **官方 DeepSeek-R1** | **800k** | ✓ | ✓ | ✓ | ✓ |
| AM | 1.4M | ✓ | ✓ | ✓ | ✓ |
| Synthetic-1 | 894k | ✓ | ✓ | ✓ | 部分 |
| Open-R1 | 220k | ✓ | ✓ | ✓ | 部分 |
| OpenThoughts | 114k | ✓ | ✓ | ✓ | 部分 |
| LIMO | **817** | ✓ | ✓ | ✓ | 部分 |

---

## 八、个人/小团队能复现什么、不能复现什么

### 8.1 ✅ Realistic：多数个人/小团队 **可以** 做到

| 目标 | 做法 | 大致成本 |
|------|------|----------|
| **本地跑 Distill 推理** | HF 权重 + Ollama/vLLM/llama.cpp | 7B：单卡 8～16GB；32B：双卡或 INT4 |
| **调用 R1 API 做应用** | platform.deepseek.com | 按量付费 |
| **复现「蒸馏路线 B」** | Open-R1 / OpenThoughts 数据 + LoRA/全参 SFT 7B | 1～4× consumer GPU，数天 |
| **自建小规模蒸馏数据** | 自有 prompt → R1 API → Math Verify 过滤 → SFT | API 费 + 单卡训练 |
| **体验 R1-Zero 涌现** | TinyZero / veRL 玩具任务 | **<$50**，1～2 卡 |
| **蒸馏后再 RL（路线 D）** | 基于官方 Distill-1.5B/7B + DeepScaleR 式 GRPO | 中等 GPU，数据 40k 级 |
| **极简 SFT 实验** | LIMO 817 条 / s1K 思路 | 单卡，几小时～一天 |
| **评测对齐** | 同 temperature、max len、AIME/MATH 脚本 | 主要是 API 评测费 |

### 8.2 ⚠️ 困难但机构级仍可能

| 目标 | 瓶颈 |
|------|------|
| **对齐官方 32B/70B Distill 分数** | 缺少官方 800k；需大量 R1 API 调用来扩数据 + 多卡训练 |
| **完整 R1-Zero on 32B+** | Open-Reasoner-Zero 等需 **129k+ 步 RL**、多卡长期训练 |
| **SimpleRL 以上、Distill 未满** | 7B 上 RL 8k 仍远低于蒸馏；需更大数据与算力 |
| **从 Base 复现完整 R1（非蒸馏）** | MoE 671B RL + 多阶段 SFT；**百万 GPU 小时级** |

### 8.3 ❌ 个人/小团队 **现实上不应指望**

| 目标 | 原因 |
|------|------|
| **训练 DeepSeek-R1 本体（671B MoE）** | DeepSeek-V3 预训练 alone **~2.66M H800 GPU hours**；非个人预算 |
| **Bitwise 复现官方 800k 数据** | 未开源；完整 rejection sampling + V3 judge 管线未知细节 |
| **无 Teacher 纯小模型 RL 超越蒸馏** | 论文 Table 6：32B RL 10K steps **仍不如** Distill-32B |
| **在 0.5B 上复现 Aha** | TinyZero：**Qwen2.5-0.5B 学不会** countdown 推理 |
| **单卡 FP16 跑 70B 长上下文** | 权重 + KV 需 **100GB+** 量级 |

### 8.4 推荐个人学习路径

```
第 1 周：Ollama 跑 Distill-7B，熟悉  格式与调参
第 2 周：用 Open-R1-Math 子集 + LoRA SFT，理解蒸馏训练
第 3 周：TinyZero 或 SimpleRL 跑通 RLVR 小实验
第 4 周：可选 DeepScaleR 式「蒸馏 checkpoint + GRPO」
```

---

## 九、法律与许可

### 9.1 DeepSeek 官方声明（MIT）

[DeepSeek-R1 仓库 LICENSE](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE)：**MIT License**。

README §7 明确：

- DeepSeek-R1 系列支持 **商业使用**；
- 允许 **任意修改与衍生作品**；
- **包括用蒸馏方式训练其他 LLM**。

因此：**用 R1 或 R1 API 生成数据去训自家模型，在 MIT 层面被允许**（仍需遵守下面基座许可）。

### 9.2 蒸馏模型的 **双重许可**

Distill 权重是「DeepSeek 微调 + 基座许可」叠加：

| 模型 | DeepSeek 层 | 基座许可 | 商用要点 |
|------|-------------|----------|----------|
| Distill-Qwen-* | MIT | **Apache 2.0** | 商用友好；需保留 NOTICE |
| Distill-Llama-8B | MIT | **Llama 3.1 Community License** | 月活 **>700 万** 需向 Meta 申请；见 Meta 官方条款 |
| Distill-Llama-70B | MIT | **Llama 3.3 Community License** | 同上 |

**实践建议：**

- 商业产品优先 **Qwen 系 Distill**（Apache 2.0 路径最清晰）。
- 使用 Llama 系前阅读 [Meta Llama License](https://llama.meta.com/llama3_1/license/) 中 **Acceptable Use Policy** 与营收/用户量条款。
- **再蒸馏**：MIT 允许；但若从 **Llama 衍生** 的 Distill 再训练，仍受 Llama 许可约束。

### 9.3 社区数据集许可

Open-R1、OpenThoughts 等多为 **Apache 2.0** 或各 repo 声明；AM、Synthetic-1 等见 HuggingFace dataset card。**商用前逐份读 license**。

### 9.4 API 使用

DeepSeek Platform API 受 **服务条款** 约束（与 MIT 模型权重不同）：注意数据留存、禁止用途、调用配额。用 API **批量生成训练数据** 前请查阅 [platform.deepseek.com](https://platform.deepseek.com) 最新 ToS。

---

## 附录 A：官方 vs 社区 Benchmark 速查（7B 档）

| 模型 | AIME 2024 pass@1 | MATH-500 | 备注 |
|------|------------------|----------|------|
| 官方 R1-Distill-Qwen-7B | **55.5** | **92.8** | 标杆 |
| Open-R1-Distill-7B | ~接近官方 | ~90.6 | Step 1 完成 |
| OpenThoughts-7B | ~31.3 | ~83.2 | 数据 114k |
| SimpleRL-Zero-7B | ~36.7 | ~77.4 | 纯 RL |

---

## 附录 B：延伸阅读

| 资料 | 链接 |
|------|------|
| DeepSeek-R1 论文 | https://arxiv.org/abs/2501.12948 |
| DeepSeek-R1 GitHub | https://github.com/deepseek-ai/DeepSeek-R1 |
| DeepSeek-V3 GitHub | https://github.com/deepseek-ai/DeepSeek-V3 |
| Open-R1 项目 | https://github.com/huggingface/open-r1 |
| 100 Days After DeepSeek-R1 综述 | https://arxiv.org/abs/2505.00551 |
| TinyZero | https://github.com/Jiayi-Pan/TinyZero |
| 本仓库训练总览 | [deepseek-training-guide-zh.md](/part-07-theory/training-guide) |

---

## 附录 C：关键结论卡片

1. **蒸馏本质**：R1 当 Teacher，80 万 CoT 样本 SFT 到 Qwen/Llama，**无 RL** 即达 SOTA 级 dense 推理。
2. **蒸馏 > 小模型 RL**：同 32B 基座，蒸馏 AIME **72.6** vs 自 RL **47.0**（+25.6）。
3. **数据未开源**：个人复现靠 R1 API + Open-R1/OpenThoughts 等公开集，分数通常 **略低于** 官方。
4. **部署**：Distill 当普通 Qwen/Llama 用 vLLM/Ollama；注意 **32K context、temp 0.6、无 system prompt**。
5. **许可**：DeepSeek MIT 允许商用与再蒸馏；Qwen Apache 友好；Llama 注意 Meta 用户量条款。

---

*文档版本：2026-06-08 · 如有官方更新以 DeepSeek GitHub 与 HuggingFace model card 为准。*
