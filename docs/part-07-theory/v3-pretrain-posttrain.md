# DeepSeek-V3 预训练与后训练深度解读（面向小白）

> **文档定位**：基于 DeepSeek-V3 Technical Report（Section 4–5）、官方 [Model Algorithm Disclosure](https://cdn.deepseek.com/policies/en-US/model-algorithm-disclosure.html) 与 [Epoch AI 成本分析](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) 整理。  
> **目标读者**：零基础或刚入门 AI 的读者；文中出现的术语均附带解释。  
> **生成日期**：2026-06-08  
> **主要参考文献**：DeepSeek-V3 Technical Report (arXiv:2412.19437)

---

## 阅读导航

| 章节 | 内容 |
|------|------|
| [0. 30 秒速览](#0-30-秒速览) | 一张图看懂 V3 训练全流程 |
| [1. 预训练数据](#1-预训练数据官方披露清洗原则与未公开边界) | 数据从哪来、怎么洗、哪些没公开 |
| [2. Next-Token Prediction 训练循环](#2-next-token-prediction-训练循环详解) | batch、sequence、loss 逐项拆解 |
| [3. 14.8T tokens 是什么概念](#3-148t-tokens-是什么概念) | 换算成书/网页的直观类比 |
| [4. 上下文 32K→128K 扩展](#4-上下文扩展-32k128k-的技术) | YaRN 两阶段方案（以 V3 论文为准） |
| [5. 训练成本 Table 1 逐项解读](#5-训练成本-table-1-逐项解读) | 2664K + 119K + 5K GPU hours |
| [6. 后训练 SFT + RL 管线](#6-后训练-sft--rl-管线v3-chat-版) | V3 Chat 对齐流程（非 R1） |
| [7. 从 R1 蒸馏推理能力](#7-从-r1-蒸馏推理能力回-v3-的方法论) | 如何把「慢思考」教给 V3 |
| [8. 能力边界对比表](#8-v3-base-vs-v3-chat-vs-r1-能力边界对比表) | 三个版本该用哪个 |
| [附录 A. 术语速查](#附录-a-术语速查) | 按字母/主题索引 |

---

## 0. 30 秒速览

DeepSeek-V3 的训练可以概括为 **「先海量读书 → 再拉长记忆 → 再学对话 → 再偷师 R1」** 四步：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 1：预训练（Pre-Training）                                          │
│  14.8T tokens · 4K 上下文 · 2664K GPU hours · 产出 DeepSeek-V3-Base     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 2：上下文扩展（Context Extension）                                 │
│  YaRN · 4K→32K→128K · 119K GPU hours                                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 3：后训练（Post-Training）                                         │
│  SFT（1.5M 条指令）+ GRPO 强化学习 + R1 蒸馏 · 5K GPU hours              │
│  产出 DeepSeek-V3（Chat 版）                                             │
└─────────────────────────────────────────────────────────────────────────┘

注：DeepSeek-R1 是在 V3-Base 之上另开一条 RL 推理专精路线，本文第 6 节聚焦 V3 Chat，第 7 节讲 R1→V3 蒸馏。
```

**三个常见版本**

| 名称 | 是什么 | 典型用途 |
|------|--------|----------|
| **DeepSeek-V3-Base** | 只完成预训练 + 上下文扩展 | 科研基座、继续微调、R1 的起点 |
| **DeepSeek-V3**（Chat） | Base + SFT + RL + R1 蒸馏 | 日常对话、编程、数学、API 服务 |
| **DeepSeek-R1** | 在 V3-Base 上专精长链推理（CoT） | 竞赛级数学/代码、复杂推理 |

---

## 1. 预训练数据：官方披露、清洗原则与未公开边界

### 1.1 什么是「预训练数据」？

**预训练（Pre-training）** 是大模型训练的**第一阶段**：让模型阅读海量文本，学习「下一个词是什么」（见第 2 节）。  
**语料（Corpus）** = 用来训练的全部文本集合。DeepSeek-V3 的预训练语料规模是 **14.8T tokens**（14.8 万亿词元）。

---

### 1.2 官方披露的数据类型

DeepSeek 在 **V3 技术报告 Section 4.1** 与 **官方算法说明页** 中披露了以下信息：

#### （1）两大数据来源（算法说明页）

| 类型 | 含义 | 举例 |
|------|------|------|
| **公开数据（Public Data）** | 从互联网合法获取的公开信息 | 网页、电子书等 |
| **授权数据（Licensed Data）** | 与第三方签协议获得的专有数据集 | 未公开具体名称与比例 |

#### （2）V3-Base 的明确边界（FMTI 透明度报告补充）

DeepSeek 在 FMTI 2025 透明度报告中进一步说明：

> **DeepSeek-V3-Base 的预训练数据 exclusively 使用 plain web pages and e-books（普通网页与电子书），未主动掺入 synthetic data（合成数据）。**

「合成数据」指用 AI 模型自动生成的文本，而非人类自然书写或爬取的内容。

#### （3）相对 DeepSeek-V2 的数据配方调整（V3 论文 Section 4.1）

| 调整项 | 说明 |
|--------|------|
| **数学与编程样本比例提高** | 增强 STEM 能力 |
| **多语言覆盖扩大** | 不再局限于英语和中文 |
| **数据处理流水线优化** | 在保持多样性的同时**减少冗余** |
| **Document Packing（文档打包）** | 借鉴 Ding et al. (2024)，把多篇短文档拼进同一训练序列，提高 GPU 利用率 |
| **不做跨样本注意力掩码** | 打包后的样本之间**不**互相「看见」，避免信息泄漏 |

#### （4）Fill-in-Middle（FIM，中间填空）策略

借鉴 DeepSeek-Coder-V2 的经验，V3 预训练还加入了 **FIM**：

- **FIM**：不只看前文猜下一个词，还要根据「前缀 + 后缀」猜**中间**缺失的文本——类似做「完形填空」。
- **格式（PSM 框架）**：
  ```
  <|fim_begin|> 前缀 <|fim_hole|> 后缀 <|fim_end|> 中间内容 <|eos_token|>
  ```
- **触发比例**：约 **10%** 的训练样本使用 FIM。
- **目的**：提升代码补全、中间插入类任务，且论文称**不损害**常规 next-token 能力。

#### （5）Tokenizer（分词器）相关

- **Byte-level BPE**：字节级 BPE 分词，词表 **128K** tokens。
- 相对 V2 优化了多语言压缩效率；并引入「标点 + 换行」组合 token，训练时随机拆分以缓解 **token boundary bias**（词元边界偏差）。

---

### 1.3 数据清洗与治理原则

官方 **Model Algorithm Disclosure** 描述了通用清洗流程（适用于 DeepSeek 全系预训练）：

| 步骤 | 做什么 | 为什么 |
|------|--------|--------|
| **自动过滤** | 剔除仇恨言论、色情、暴力、垃圾信息、潜在侵权内容 | 安全与合规 |
| **去重（Deduplication）** | 文档级 + 字符串级去重（早期 DeepSeek-LLM 文档提到 MinHashLSH） | 避免模型「背答案」、浪费算力 |
| **质量过滤** | 启发式规则 + 模型打分，去掉低质量网页，保留低资源语言知识 | 提升信噪比 |
| **隐私处理** | 排除信用卡号、唯一标识符；尽力筛除个人信息 | 降低隐私风险 |
| **偏见缓解** | 算法 + 人工抽检，减轻统计偏见对模型价值观的影响 | 公平性 |

**DeepSeek-V3 论文额外强调**：流水线 refined to **minimize redundancy while maintaining corpus diversity**——在「多样」与「不重复」之间找平衡。

---

### 1.4 未公开部分与信息边界（读者应知晓）

以下信息 **官方未充分披露**，第三方只能推断：

| 未公开项 | 已知边界 | 未知/模糊部分 |
|----------|----------|---------------|
| **各数据源占比** | 数学/代码比例「相对 V2 提高」 | 网页 vs 电子书 vs 授权数据的具体 % |
| **爬虫细节** | 早期 DeepSeek-LLM 提到 respect robots.txt | V3 专用爬虫名称、爬取域名列表 |
| **授权数据集清单** | 存在 licensed data | 具体供应商、授权范围 |
| **英文/中文/其他语言比例** | 英中占「majority（大多数）」 | 精确语言分布 |
| **合成数据** | V3-Base **未主动使用** | 网页中可能**间接**含其他模型（如 OpenAI）生成内容 |
| **Cooldown 阶段数据** | 最后 500B tokens 学习率更低 | 是否混入额外 curated 数据未详述 |

> **重要区分**：V3-Base 预训练 **不用** 合成数据；但 **V3 Chat 后训练** 会用到 R1 等模型**生成**的推理数据——那是后训练阶段，不是预训练语料。

---

## 2. Next-Token Prediction 训练循环详解

### 2.1 核心任务：猜下一个词

**Next-Token Prediction（下一词元预测）** 是自回归语言模型的标准预训练目标：

1. 输入一段文本：`今天天气很`
2. 模型输出每个可能下一个词的概率：`好` 0.4、`热` 0.3、`糟` 0.1 …
3. 与真实下一个词 `好` 对比，计算 **Loss（损失）**
4. 通过 **反向传播（Backpropagation）** 微调参数，让 Loss 变小

这叫 **自监督学习（Self-supervised Learning）**：标签就是文本本身的下一个词，无需人工标注。

---

### 2.2 一条训练样本长什么样？

预训练阶段关键超参（V3 论文 Section 4.2）：

| 概念 | V3 取值 | 解释 |
|------|---------|------|
| **Sequence Length（序列长度）** | **4096 tokens（4K）** | 每条训练样本最多含 4096 个 token；更长文档会被截断或打包 |
| **Batch Size（批大小）** | **3072 → 15360** | 见下文「Batch 调度」 |
| **Training Tokens** | **14.8T** | 整个预训练阶段模型「看过」的 token 总量 |

**Document Packing 示例**（概念图）：

```
一条 4096 token 的训练序列可能打包了 3 篇短文章：
[文章A tokens | 文章B tokens | 文章C tokens | padding...]
         ↑              ↑              ↑
    样本内可见      样本内可见      样本内可见
    但 A/B/C 之间用 attention mask 隔离，互不可见（V3 预训练不做跨样本 mask 的说法需准确理解：
    论文原文是 pre-training 阶段 "do not incorporate cross-sample attention masking"——
    即打包后不额外做跨样本掩码，与部分其他工作不同；SFT 阶段则相反，会加 sample masking）
```

> **注**：V3 **预训练**明确 **不** 使用 cross-sample attention masking；**后训练 SFT** 则 **会** 使用 sample masking 防止样本串台。两处策略不同，勿混淆。

---

### 2.3 Batch（批大小）调度

**Batch Size** = 每次参数更新前，模型并行处理的 **序列条数**（不是 token 总数）。

V3 采用 **Batch Size Scheduling（批大小调度）**：

| 训练进度 | Batch Size | 含义 |
|----------|------------|------|
| 前 **469B（4690 亿）** tokens | 从 **3072 线性增至 15360** | 训练初期用小 batch 更稳定，后期用大 batch 提高吞吐 |
| 剩余训练 | 固定 **15360** | 维持最大吞吐 |

**Token 级别的有效 batch** ≈ `batch_size × sequence_length`。  
例如在 batch=15360、seq=4096 时，每步约处理 **6290 万 tokens**（理论峰值，实际因 packing 会有波动）。

---

### 2.4 Loss（损失函数）构成

V3 的预训练目标 = **主损失 + MTP 辅助损失**。

#### （1）主损失：标准 Cross-Entropy（交叉熵）

对每个位置 \(i\)，模型预测下一个 token 的概率分布，与真实 token \(x_i\) 算交叉熵：

\[
\mathcal{L}_{\text{main}} = -\frac{1}{T}\sum_{i=1}^{T} \log P(x_i \mid x_{<i})
\]

- **Cross-Entropy（交叉熵）**：衡量「预测分布」与「真实 one-hot 标签」的差距；越小越好。
- **Softmax**：把模型输出的 logits 转成概率。

#### （2）辅助损失：Multi-Token Prediction（MTP，多词元预测）

V3 在猜 **下一个词** 之外，还训练模型猜 **下下一个词**（MTP depth = 1）：

| 项 | 说明 |
|----|------|
| **MTP depth** | 1（共预测 2 个未来 token：next1 + next2） |
| **MTP Loss 权重 \(\lambda\)** | 前 10T tokens：**0.3**；剩余 4.8T tokens：**0.1** |
| **推理时** | MTP 模块可**丢弃**，不影响主模型部署；也可用于 speculative decoding 加速 |

总损失概念上为：

\[
\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda \cdot \mathcal{L}_{\text{MTP}}
\]

**为什么加 MTP？** 论文 ablation（Table 4）显示：在相同推理成本下，MTP  consistently 提升 MMLU、HumanEval、GSM8K 等 benchmark。

---

### 2.5 一次完整 Training Step（训练步）发生了什么？

用「伪代码 + 白话」描述一步：

```
1. 从数据管道取一批序列（batch_size 条，每条最长 4096 tokens）
2. Forward（前向传播）
   - Token Embedding → 61 层 Transformer（含 MoE）→ Output Head → logits
   - MoE：每个 token 激活 8 个 routed expert + 1 个 shared expert（共 37B 激活参数）
3. 计算 Loss
   - L_main：每个位置的 next-token 交叉熵
   - L_MTP：第二个未来 token 的交叉熵
   - L_total = L_main + λ * L_MTP
4. Backward（反向传播）
   - 计算梯度，Gradient Clipping（梯度裁剪，max norm = 1.0）
5. Optimizer Update
   - AdamW（β1=0.9, β2=0.95, weight_decay=0.1）更新参数
6. （可选）更新 MoE 负载均衡 bias；更新 EMA 参数副本
```

**并行策略**（Infrastructure Section 3）：16-way Pipeline Parallel + 64-way Expert Parallel + ZeRO-1 Data Parallel，2048 张 H800，**不用 Tensor Parallelism**。

**精度**：FP8 混合精度训练——核心 GEMM 用 FP8，Embedding/Attention/MoE gating 等保留 BF16/FP32 以保稳定。

---

### 2.6 学习率（Learning Rate）调度一览

| 阶段 | Token 范围 / 步数 | 学习率 |
|------|-------------------|--------|
| Warmup | 前 2000 steps | 0 → **2.2×10⁻⁴** 线性升温 |
| 恒定 | 至 10T tokens | **2.2×10⁻⁴** |
| Cosine Decay | 接下来 4.3T tokens | **2.2×10⁻⁴ → 2.2×10⁻⁵** |
| Cooldown 1 | 最后 500B 的前 333B | **2.2×10⁻⁵** 恒定 |
| Cooldown 2 | 最后 167B | **7.3×10⁻⁶** 恒定 |

**Cooldown（冷却阶段）**：训练末期降低学习率，让模型「细读」最后一批数据，巩固知识。

---

## 3. 14.8T tokens 是什么概念？

### 3.1 先理解 Token

**Token（词元）** 是模型读文字的最小单位，不等于「一个字」或「一个词」：

| 语言 | 粗略换算 |
|------|----------|
| 英文 | 1 词 ≈ 1 token |
| 中文 | 1 字 ≈ 1–2 tokens |

**14.8T = 14.8 trillion = 14,800,000,000,000（14.8 万亿）tokens**。

---

### 3.2 换成「书」的量级

| 类比对象 | 估算 | 与 14.8T 的关系 |
|----------|------|-----------------|
| **一本流行小说** | ~10 万字（中文）≈ 12–15 万 tokens | ≈ **1 亿本书** 量级（粗算） |
| **一本学术专著** | ~30 万字 ≈ 40 万 tokens | ≈ **3700 万册** |
| **美国国会图书馆藏书** | ~4000 万册 | 14.8T ≈ **把整个国会图书馆的书读 ~250 遍**（按平均 25 万 tokens/本粗算） |
| **《哈利·波特》全集 7 册** | ~100 万英文词 ≈ 100 万 tokens | ≈ **1480 万套** |

> 以上为 **数量级直觉**，非精确统计；实际语料以网页短文本为主，单条远短于一本书。

---

### 3.3 换成「网页」的量级

| 类比 | 估算 |
|------|------|
| **普通新闻网页** | ~800–1500 英文词 ≈ 1000–2000 tokens |
| **14.8T tokens** | ≈ **70 亿–150 亿个网页**（取中值约 **100 亿页**） |
| **中文互联网公开索引** | 全网页面数因搜索引擎而异，14.8T 相当于把主流中文+英文公开文本**多轮去重后**的累积阅读量 |

---

### 3.4 换成「训练时间」

论文给出：**每 1T tokens ≈ 180K H800 GPU hours ≈ 3.7 天**（2048 卡集群）。

| 量 | 结果 |
|----|------|
| 14.8T tokens | 14.8 × 180K = **2,664,000 GPU hours** |
| 日历时间 | 约 **55 天**（2,664K / 2048 / 24） |
| 美元成本（$2/GPU hour） | **≈ $533 万** |

Epoch AI 独立验证：14.8 × 180,000 ≈ 2.66M GPU hours，与官方 Table 1 的 **2664K** 一致。

---

### 3.5 横向对比其他模型

| 模型 | 预训练 tokens | 备注 |
|------|---------------|------|
| DeepSeek-V3 | **14.8T** | 本模型 |
| Qwen2.5 | **18T** | 比 V3 多 ~20%，但 V3 Chat 中文 SimpleQA 仍更强（论文 Section 5.3.2） |
| LLaMA 3.1 405B | ~15T+ | 规模相近 |
| DeepSeek-V2 | 8.1T | V3 约 1.8× |

---

## 4. 上下文扩展 32K→128K 的技术

### 4.1 为什么预训练只有 4K，最终却支持 128K？

**Context Length（上下文长度）** = 模型一次能「看见」的最大 token 数。

| 阶段 | 最大上下文 | 原因 |
|------|------------|------|
| 预训练 | **4K** | 算力与内存：注意力复杂度随长度平方增长 |
| 上下文扩展 | **32K → 128K** | 专门微调阶段，Teaching 模型读长文 |
| SFT 后 | **128K**（Chat 默认 8K 输出限制另论） | NIAH 测试验证到 128K |

预训练教「知识」；**Context Extension** 教「长距离记忆与检索」。

---

### 4.2 V3 用什么方法？—— YaRN（论文明确采用）

V3 论文 Section 4.3 写明：

> After pre-training, we apply **YaRN** for context extension … two additional training phases, each comprising **1000 steps**.

**YaRN（Yet another RoPE extensioN）** 是一种 **RoPE（Rotary Position Embedding，旋转位置编码）扩展** 方法：

| 术语 | 解释 |
|------|------|
| **RoPE** | 给每个 token 位置编码「序号」，让注意力机制知道谁在前谁在后 |
| **YaRN** | 修改 RoPE 的频率分布，使在**不重新预训练**的情况下 extrapolate（外推）到更长序列 |
| **NTK-aware scaling** | 另一种常见扩展思路；**V3 论文未采用 NTK，而是 YaRN**——本文以论文为准 |

**V3 的 YaRN 配置**（与 DeepSeek-V2 一致）：

| 参数 | 值 | 作用（直觉） |
|------|-----|--------------|
| 应用对象 | 仅 **decoupled shared key \(k^R\)** | 只改 Key 的 RoPE，减少干扰 |
| scale \(s\) | **40** | 目标扩展倍数相关 |
| \(\beta\) | **1** | YaRN 插值参数 |
| \(\alpha\) | **32** | YaRN 高频/低频分割 |
| scaling factor \(\sqrt{t}\) | **0.1 ln(s) + 1** | 注意力温度缩放 |

---

### 4.3 两阶段扩展训练

| 阶段 | 目标长度 | 训练 steps | Batch Size | 学习率 |
|------|----------|------------|------------|--------|
| **Phase 1** | 4K → **32K** | 1000 | **1920** | **7.3×10⁻⁶** |
| **Phase 2** | 32K → **128K** | 1000 | **480**（因内存降低） | **7.3×10⁻⁶** |

- 学习率与预训练 **Cooldown 末期** 一致，避免「冲垮」已学知识。
- Phase 2 batch 变小：128K 序列极占显存，只能减少并行条数。

**验证**：Figure 8「Needle In A Haystack（NIAH）」—— 在 128K 长文中插入隐藏事实，SFT 后的 V3 在各长度上检索稳定。

---

### 4.4 YaRN vs NTK：小白该记住什么？

| 方法 | V3 是否使用 | 一句话 |
|------|-------------|--------|
| **YaRN** | ✅ 使用 | 插值 + 缩放 RoPE 频率，两阶段 1000 step 微调 |
| **NTK** | ❌ 论文未提 | 社区常用另一种 RoPE 缩放；V3 报告以 YaRN 为准 |
| **继续预训练长文** | ❌ 非主路径 | V3 选择「短预训练 + 专用扩展阶段」，更省算力 |

---

## 5. 训练成本 Table 1 逐项解读

### 5.1 官方 Table 1 原文

| 阶段 | H800 GPU Hours | 美元（$2/GPU hour） |
|------|----------------|---------------------|
| **Pre-Training** | **2664K** | **$5.328M** |
| **Context Extension** | **119K** | **$0.238M** |
| **Post-Training** | **5K** | **$0.01M** |
| **Total** | **2788K** | **$5.576M** |

> 论文注明：以上**仅含正式训练 run**，**不含**架构/算法/数据消融实验与前期研究成本。

---

### 5.2 2664K GPU Hours —— 预训练

| 拆解项 | 数值 | 解读 |
|--------|------|------|
| 总 tokens | 14.8T | 语料规模 |
| 每 T tokens 成本 | **180K GPU hours** | 工程优化后（FP8、DualPipe、MoE 通信重叠） |
| 验算 | 14.8 × 180K = **2,664,000** | 与 2664K 吻合 |
| 集群 | **2048 × H800** | 每节点 8 卡，NVLink 机内 + InfiniBand 机间 |
| 日历时间 | 2664K ÷ 2048 ÷ 24 ≈ **54 天** | 约 **不到 2 个月** |
| 稳定性 | **零不可恢复 loss spike** | 无需 rollback checkpoint |

**Epoch AI 独立分析要点**：

- 算术量约 **3×10²⁴ FLOP**（6 × 37B active × 14.8T）
-  implied **MFU（Model FLOP Utilization）≈ 23%**—— 对 MoE 大模型属合理；通信开销大，非「浪费」
- 结论：**官方数字可信，甚至偏保守**； mystery 不是「为何这么便宜」而是 MoE 训练本就难拉高 MFU

---

### 5.3 119K GPU Hours —— 上下文扩展

| 项 | 说明 |
|----|------|
| 占比 | 119K / 2788K ≈ **4.3%** 总算力 |
| 内容 | YaRN 两阶段 × 各 1000 steps；32K batch 1920 + 128K batch 480 |
| 产出 | Base 模型可处理 **128K** 输入 |
| 成本 | **≈ $23.8 万** |

直觉：相对 14.8T 预训练，长上下文微调是「小修」，但对用户可用性（读长 PDF、长代码库）至关重要。

---

### 5.4 5K GPU Hours —— 后训练（SFT + RL + R1 蒸馏）

| 项 | 说明 |
|----|------|
| 占比 | 5K / 2788K ≈ **0.18%** 总算力 |
| 内容 | 1.5M 条 SFT + GRPO RL + 从 R1 蒸馏推理数据 |
| 产出 | **DeepSeek-V3 Chat** |
| 成本 | **≈ $1 万** |
| GitHub README 表述 | Post-Training + Distillation 合计 **0.1M GPU hours**—— 与 Table 1 的 5K（仅 post-training 一行）口径略有出入；**以 Table 1 分项为准** |

**为什么后训练这么便宜？**

1. **数据量小**：1.5M 样本 vs 14.8T tokens，差 6–7 个数量级  
2. **Epoch 数少**：SFT 仅 2 epochs  
3. **基座已强**：V3-Base 已具备知识，后训练只做「对齐与专项增强」

---

### 5.5 三阶段成本饼图（直觉）

```
预训练 2664K  ████████████████████████████████████████  95.6%
上下文  119K  ██                                         4.3%
后训练    5K  ▏                                         0.2%
```

**关键洞察**：把 Base 训好（2664K）是「大头」；Chat 体验（5K）是「点睛」—— 但 R1 蒸馏带来的 AIME 等提升巨大，**ROI 极高**。

---

## 6. 后训练 SFT + RL 管线（V3 Chat 版）

> 本节描述 **DeepSeek-V3 Chat** 的后训练，**不是** DeepSeek-R1 的 RL 推理训练。R1 是在 V3-Base 上另起炉灶的专精路线（见第 7 节边界说明）。

### 6.1 后训练是什么？

| 阶段 | 英文 | 目标 |
|------|------|------|
| **SFT** | Supervised Fine-Tuning，监督微调 | 学「听指令、按格式回答」 |
| **RL** | Reinforcement Learning，强化学习 | 学「什么回答更好、更符合人类偏好」 |

预训练模型像「读了很多书但没上过礼仪课」；后训练像「客服话术 + 用户满意度培训」。

---

### 6.2 数据：1.5M 条指令实例

V3 论文 Section 5.1 披露 SFT 数据集 **1.5M instances**，分两大类：

#### A. 推理类（Reasoning Data）

| 项 | 说明 |
|----|------|
| 领域 | 数学、编程竞赛题、逻辑谜题 |
| 来源 | 内部 **DeepSeek-R1** 模型生成 + 专家模型 pipeline 筛选 |
| 痛点 | R1 生成内容准确但 **overthinking（想太多）、格式差、过长** |
| 目标 | 平衡 **R1 高准确率** 与 **简洁可读格式** |

#### B. 非推理类（Non-Reasoning Data）

| 项 | 说明 |
|----|------|
| 领域 | 创意写作、角色扮演、简单问答 |
| 来源 | **DeepSeek-V2.5** 生成 + **人工审核** |

---

### 6.3 SFT 训练设置

| 超参 | 值 |
|------|-----|
| 基座 | **DeepSeek-V3-Base**（已完成上下文扩展） |
| Epochs | **2** |
| 学习率 | Cosine：**5×10⁻⁶ → 1×10⁻⁶** |
| Sequence Packing | 多条样本拼一条长序列 |
| **Sample Masking** | ✅ 样本间 **互不可见**（与预训练相反） |

---

### 6.4 RL：GRPO + 双 Reward Model

#### （1）GRPO（Group Relative Policy Optimization）

| 术语 | 解释 |
|------|------|
| **PPO** | 经典 RL 算法，需一个与 policy 同尺寸的 **Critic（评论家）** 模型，很贵 |
| **GRPO** | DeepSeek 提出：对同一问题采样 **一组** 回答，用组内相对得分当 baseline，**省掉 Critic** |

流程：

1. 对每个 prompt 采样 \(G\) 个回答  
2. 每个回答得 reward \(r_i\)  
3. Advantage = \((r_i - \text{mean}) / \text{std}\)  
4. 优化 policy，使高 reward 回答更可能被生成  

#### （2）Reward Model（奖励模型）双轨制

| 类型 | 适用场景 | 例子 |
|------|----------|------|
| **Rule-Based RM** | 可自动判对错 | 数学题 `\boxed{}` 答案、LeetCode 测例 |
| **Model-Based RM** | 开放题、写作 | 从 V3 SFT checkpoint 训练；输入含 **CoT 式打分解释** 防 reward hacking |

#### （3）Prompt 多样性

RL 阶段 prompt 覆盖：**coding、math、writing、role-play、QA** —— 既对齐人类偏好，也补 SFT 数据覆盖不到的 benchmark。

#### （4）Self-Rewarding（自奖励）

对难以硬编码的场景，用 **Constitutional AI** 思路：让 **DeepSeek-V3 自己投票** 当 feedback，增强主观题对齐（Section 5.4.2）。

---

### 6.5 V3 Chat 后训练管线总图

```
DeepSeek-V3-Base（128K）
        │
        ├─► 构建 1.5M SFT 数据
        │     ├─ 推理类：R1 + 专家模型 + rejection sampling
        │     └─ 非推理类：V2.5 + 人工校验
        │
        ▼
   SFT（2 epochs, LR 5e-6→1e-6）
        │
        ▼
   GRPO RL（rule RM + model RM + 多域 prompt）
        │
        ▼
   （可选）Constitutional AI 自奖励迭代
        │
        ▼
   DeepSeek-V3 Chat  ← 第 7 节：此阶段还融入 R1 蒸馏数据
```

---

## 7. 从 R1 蒸馏推理能力回 V3 的方法论

### 7.1 为什么要「蒸馏」？

| 概念 | 解释 |
|------|------|
| **DeepSeek-R1** | 在 V3-Base 上通过 **长链 CoT + RL** 练出的「慢思考」推理模型 |
| **Knowledge Distillation（知识蒸馏）** | 让「学生模型」学习「教师模型」的输出模式，而不只是学标准答案 |
| **为何 V3 Chat 要蒸馏 R1？** | R1 数学/代码强，但 **太慢、太长、不适合日常对话**；蒸馏把 verification/reflection 模式「压缩」进 V3 Chat |

> **注意方向**：这里是 **R1 → V3 Chat**（把推理模式灌进通用对话模型），**不是** R1-Distill-Qwen（那是 R1 → 小模型，另产品线）。

---

### 7.2 官方方法论（Section 5.1 + 5.4.1）

#### Step 1：训练领域「专家模型（Expert Model）」

对每个领域（代码 / 数学 / 通用推理）：

```
Expert = SFT + RL 迭代训练出的中间 checkpoint
```

#### Step 2：双格式 SFT 样本

对每个问题生成 **两种** 训练样本：

| 格式 | 内容 |
|------|------|
| **格式 A** | `问题 + 原始简洁回答` |
| **格式 B** | `系统提示（含 reflection/verification 指令）+ 问题 + R1 长回答` |

系统提示引导模型学习 **反思、验证** 模式。

#### Step 3：Expert 上跑 RL

- **高温采样** 生成融合 R1 与原始数据风格的回答  
- 数百步 RL 后，Expert 即使无 system prompt 也会 **战略性** 融入 R1 模式  

#### Step 4：Rejection Sampling（拒绝采样）

- 用 Expert 批量生成候选回答  
- **筛选高质量** 样本 → 构成最终 V3 SFT 数据  
- 保留 R1 **正确性**，控制 **长度与格式**

#### Step 5：平衡准确率 vs 长度

Table 9 ablation（在 V2.5 上验证）：

| 设置 | LiveCodeBench Pass@1 | 平均输出长度 | MATH-500 Pass@1 | 平均长度 |
|------|------------------------|--------------|-----------------|----------|
| Baseline（短 CoT） | 31.1 | 718 | 74.6 | 769 |
| **+R1 Distill** | **37.4** | 783 | **83.2** | 1510 |

蒸馏 **显著提升** 推理 benchmark，但 **输出变长 ~2×**。V3 最终 ** carefully selected optimal settings** 在准确率与效率间折中。

---

### 7.3 与 DeepSeek-R1 训练的关系（避免混淆）

| 路线 | 起点 | 核心 | 产物 |
|------|------|------|------|
| **V3 Chat 后训练** | V3-Base | SFT + GRPO + **消费** R1 生成数据 | DeepSeek-V3 |
| **R1 训练** | V3-Base | R1-Zero GRPO + Cold Start + 二阶段 RL + 800K reasoning SFT | DeepSeek-R1 |

Epoch AI 估算：**R1 的 RL 阶段 GPU 成本 ~$100 万**，叠加在 V3 预训练 ~$530 万之上—— 但 R1 是 **独立产品**，不是 V3 Table 1 里那 5K hours 的一部分。

**数据流向**：

```
V3-Base ──► R1 训练 ──► DeepSeek-R1（教师）
                │
                └──► 生成推理数据 ──► 蒸馏回 V3 Chat（学生）
```

---

## 8. V3-Base vs V3-Chat vs R1 能力边界对比表

### 8.1 定位一句话

| 模型 | 一句话定位 |
|------|------------|
| **V3-Base** | 最强开源 **基座**：知识广，不会聊天，无安全对齐 |
| **V3 Chat** | 最强开源 **通用对话**：日常任务 + 较强数学代码，**非** 纯推理专精 |
| **R1** | **推理专精**：AIME/Codeforces 级，输出长 CoT，适合「难题」 |

---

### 8.2 训练与部署对比

| 维度 | V3-Base | V3 Chat | DeepSeek-R1 |
|------|---------|---------|-------------|
| **训练阶段** | 预训练 + 上下文扩展 | + SFT + RL + R1 蒸馏 | V3-Base + 多阶段 RL + reasoning SFT |
| **GPU Hours（官方）** | ~2778K（含扩展） | +5K | 未单独披露（Epoch AI 估 ~+$100 万 RL） |
| **是否对话对齐** | ❌ | ✅ | ✅（偏推理风格） |
| **典型输出** | 续写文本 | 结构化回答 | 长 Chain-of-Thought |
| **上下文** | 128K | 128K（评测常限 8K 输出） | 128K（R1 评测 max gen 32K） |
| **激活参数** | 37B | 37B | 37B |

---

### 8.3 Benchmark 能力对比（精选）

> 数据来源：V3 论文 Table 3/6；R1 论文/README Table。评测设置可能不同，**仅作边界直觉**。

| Benchmark | V3-Base | V3 Chat | R1 | 解读 |
|-----------|---------|---------|-----|------|
| **MMLU** | 87.1 | 88.5 | ~90+ | Chat 略升；R1 综合知识更强 |
| **MATH-500** | 61.6 | **90.2** | **97.3** | 蒸馏+R1 数据让 Chat 暴涨；R1 仍更高 |
| **AIME 2024** | ~低 | **39.2** | **79.8** | R1 竞赛级；Chat 中等偏上 |
| **HumanEval** | 65.2 | 82.6 (Mul) | ~79+ | Chat 代码生成很强 |
| **LiveCodeBench** | 19.4 | **40.5** (CoT) | ~65+ | R1 代码推理领先 |
| **Codeforces Rating** | — | 51.6 percentile | **2029** | R1 达专家级 |
| **SWE-Bench Verified** | — | **50.8** | ~49.2 | Chat 工程任务已很强 |
| **IFEval（指令遵循）** | — | **86.5** | ~83 | Chat 专精指令 |
| **Arena-Hard** | — | **85.5** | ~92+ | R1 开放对话 judge 更高 |
| **SimpleQA（事实）** | — | 38.2 | ~30 | V3 非 R1 强项；R1 牺牲部分事实换推理 |

---

### 8.4 选型建议（给应用开发者）

| 你的需求 | 推荐 |
|----------|------|
| 继续预训练 / 领域微调 | **V3-Base** |
| 通用 API、聊天、编程助手 | **V3 Chat** |
| 奥数、竞赛编程、复杂证明 | **R1** |
| 边缘设备 / 小模型 | **R1-Distill-***（蒸馏到小模型，非本文主体） |

---

### 8.5 能力边界「硬边界」总结

1. **V3-Base** 强在 **知识广度与基座 benchmark**，**不会** follow 指令，**无** RLHF 安全层。  
2. **V3 Chat** 是 **通用最优开源**；数学代码 **远强于 Base**，但 **弱于 R1** 的极限推理；蒸馏带来 **更长输出**。  
3. **R1** 是 **o1 类推理模型**；日常闲聊、短问答 **不如 V3 Chat 经济**；SimpleQA 等事实题 **未必更好**。  
4. **三者共享同一架构**（671B MoE / 37B active / MLA）；差异主要在 **后训练配方与推理时 decoding 策略**。

---

## 附录 A. 术语速查

| 术语 | 英文 | 解释 |
|------|------|------|
| 词元 | Token | 模型处理文本的最小单位 |
| 预训练 | Pre-training | 海量 next-token 学习阶段 |
| 监督微调 | SFT | 用「输入-输出」示范教指令跟随 |
| 强化学习 | RL | 用 reward 信号优化生成策略 |
| 混合专家 | MoE | 每 token 只激活部分子网络，省算力 |
| 交叉熵 | Cross-Entropy | 分类/预测任务常用损失函数 |
| 批大小 | Batch Size | 每次更新看的序列条数 |
| 序列长度 | Sequence Length | 单条样本最大 token 数 |
| 上下文窗口 | Context Window | 模型能 attend 的最大历史长度 |
| 梯度裁剪 | Gradient Clipping | 限制梯度范数防训练爆炸 |
| 知识蒸馏 | Distillation | 大模型教小模型或强模式教通用模型 |
| 思维链 | CoT | Chain-of-Thought，逐步推理文本 |
| 拒绝采样 | Rejection Sampling | 生成多个候选，只保留高质量 |
| 位置编码扩展 | RoPE Extension | 让模型适应更长序列的技术总称 |
| YaRN | Yet another RoPE extensioN | V3 采用的 RoPE 扩展算法 |
| GRPO | Group Relative Policy Optimization | 无 Critic 的 RL 算法 |
| FP8 | 8-bit Floating Point | 低精度浮点，省显存与带宽 |
| MFU | Model FLOP Utilization | 实际算力利用率 |
| GPU Hour | GPU 小时 | 1 块 GPU 跑 1 小时 = 1 GPU hour |

---

## 附录 B. 参考文献与链接

| 资料 | 链接 |
|------|------|
| DeepSeek-V3 Technical Report | https://arxiv.org/abs/2412.19437 |
| DeepSeek 官方算法说明 | https://cdn.deepseek.com/policies/en-US/model-algorithm-disclosure.html |
| Epoch AI：R1 训练成本分析 | https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1 |
| DeepSeek-V3 GitHub | https://github.com/deepseek-ai/DeepSeek-V3 |
| DeepSeek-R1 Technical Report | https://arxiv.org/abs/2501.12948 |
| FMTI DeepSeek 透明度报告 | https://crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html |

---

## 附录 C. 文档元信息

- **主要章节数**：8 + 3 附录  
- **覆盖论文章节**：V3 Report §4（Pre-Training）、§5（Post-Training）、§3（Infrastructure 节选）、Table 1  
- **版本说明**：以 DeepSeek-V3 初版技术报告（2024-12）为准；R1 为 2025-01 发布  

---

*本文档供学习研究使用；模型能力与价格随版本更新可能变化，请以 DeepSeek 官方最新发布为准。*
