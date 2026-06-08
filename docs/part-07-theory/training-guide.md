# DeepSeek 训练全过程：从零开始的超详细解读

> 本文基于 DeepSeek 官方技术报告、算法说明与 R1 论文整理。  
> 目标：即使从未接触过 AI，也能顺着读下来，理解「DeepSeek 是怎么被训练出来的」。  
> 生成日期：2026-06-08

---

## 一、先用一个比喻建立直觉

把训练 DeepSeek 想象成 **培养一位超级学霸**：

| 人生阶段 | 对应 AI 训练阶段 | 在做什么 |
|---------|------------------|---------|
| 0～18 岁大量读书 | **预训练（Pre-training）** | 读海量书/网页，学语言、常识、逻辑 |
| 高考冲刺班 | **监督微调（SFT）** | 老师给标准答案，学「怎么答题、怎么听指令」 |
| 反复模考 + 改错 | **强化学习（RL）** | 自己做题，对了加分、错了扣分，越练越强 |
| 专攻奥数 | **R1 推理训练** | 不只背答案，而是练「长时间思考、自我检查」 |

DeepSeek 不是单一模型，而是一 **产品家族**：

- **DeepSeek-V3**：通用对话基座（像「全科状元」）
- **DeepSeek-R1**：推理专精（像「奥数金牌 + 编程竞赛选手」）
- **DeepSeek-R1-Distill-***：把 R1 的能力「蒸馏」到小模型上（像把学霸笔记复印给普通学生）

下面每个术语都会解释，再讲完整训练流水线。

---

## 二、核心名词解释（词典）

读正文前，可先扫一遍；遇到不懂的词也可回来查。

### 2.1 基础概念

**大语言模型（LLM, Large Language Model）**  
能理解和生成人类语言的神经网络。输入文字，输出文字。ChatGPT、DeepSeek、Claude 都属于 LLM。

**神经网络（Neural Network）**  
模仿大脑神经元连接的计算结构。多层「节点」叠加，通过大量数据调整内部「旋钮」（参数），学会完成任务。

**参数（Parameters / 权重 Weights）**  
模型内部可学习的数字，总量决定模型「容量」。  
- DeepSeek-V3 总参数约 **6710 亿（671B）**  
- 但每次处理一个词，实际只激活约 **370 亿（37B）**（见 MoE）

**Token（词元）**  
模型处理文字的最小单位。英文约 1 词 ≈ 1 token；中文约 1 字 ≈ 1～2 token。  
「14.8 万亿 token」≈ 读了极海量文字。

**Transformer（变换器架构）**  
2017 年 Google 提出的结构，是当前几乎所有 LLM 的基础。核心是 **自注意力（Self-Attention）**：读一句话时，每个词都能「看到」其他词，理解上下文关系。

**基座模型（Base Model）**  
只完成预训练、尚未针对聊天/指令优化的模型。  
DeepSeek-V3-Base 就是 R1 的起点。

**Chat 模型 / 对齐模型**  
经过 SFT + RL 后，能按人类指令对话、更安全有用的版本。  
DeepSeek-V3（对话版）、DeepSeek-R1 都属于这类。

### 2.2 训练相关

**预训练（Pre-training）**  
第一阶段。模型读海量文本，任务通常是：**给定前文，预测下一个词**（Next Token Prediction）。  
类比：婴儿通过听人说话学语言规律，没人逐字教，但听多了就会说。

**自监督学习（Self-supervised Learning）**  
标签来自数据本身（下一个词就是「标准答案」），不需要人工标注每一句。

**损失函数（Loss Function）**  
衡量「模型猜错了多少」的分数。训练目标：**让 Loss 越来越小**。

**梯度下降（Gradient Descent）**  
根据 Loss 告诉模型「参数该往哪调」，一步步优化。  
**反向传播（Backpropagation）**：从输出往回算，更新每一层参数。

**Loss Spike（损失尖峰）**  
训练中途 Loss 突然暴涨，模型可能「学歪」。严重时要 **回滚（Rollback）** 到之前 checkpoint。  
DeepSeek-V3 报告：全程 **没有** 不可恢复尖峰，训练非常稳定。

**Checkpoint（检查点）**  
训练中间保存的模型快照，可恢复或继续训练。

**微调（Fine-tuning）**  
在已预训练模型上，用更小、更专门的数据继续训练。

**监督微调（SFT, Supervised Fine-tuning）**  
用「问题 → 标准答案」配对数据教模型：  
- 用户问什么  
- 助手该怎么答  
- 用什么格式（如 Markdown、分步骤）

**强化学习（RL, Reinforcement Learning）**  
智能体做动作 → 环境给 **奖励（Reward）** 或惩罚 → 调整策略使长期奖励最大。  
LLM 里：模型生成回答 → 评分系统打分 → 用 GRPO 等算法更新模型。

**RLHF（Reinforcement Learning from Human Feedback）**  
用人类偏好训练 **奖励模型（Reward Model）**，再用 RL 优化 LLM。ChatGPT 早期典型路径。

**RLVR（RL with Verifiable Rewards）**  
用 **可自动验证** 的规则打分（数学题答案对不对、代码能不能跑通），不依赖人类逐条打分。DeepSeek-R1 推理阶段的核心。

**奖励黑客（Reward Hacking）**  
模型找到「骗过评分系统拿高分」的歪门邪道，而非真学会解题。  
DeepSeek 在 R1-Zero 中 **刻意不用** 神经网络奖励模型， partly 为避免此问题。

**蒸馏（Distillation）**  
大模型生成高质量数据/行为，用来训练小模型。  
DeepSeek 用 R1 的推理数据训练 7B、14B 等小模型，效果往往优于在小模型上直接做 RL。

**拒绝采样（Rejection Sampling）**  
同一问题让模型答多次，**只保留答对/质量高的**，丢弃差的。用于构造 SFT 数据。

**Cold Start（冷启动）**  
RL 开始前，用少量高质量 SFT 数据先「热机」，避免从 raw base 直接 RL 时不稳定、输出难读。

### 2.3 推理与思维链

**推理（Reasoning）**  
多步逻辑推导：数学证明、代码调试、复杂分析。不能一步猜答案。

**思维链（CoT, Chain-of-Thought）**  
模型在给出最终答案前，先写出中间推理步骤。  
例：「先设 x=…，再代入…，所以答案是 42」。

**长 CoT（Long CoT）**  
推理步骤很长，可能几百到几千 token。OpenAI o1、DeepSeek-R1 的特点。

**Test-time Scaling（推理时扩展）**  
不增大模型，而是 **生成更多思考 token、花更长时间想**，提升难题正确率。

**Reflection（反思）**  
模型中途发现前面错了，说「等等，我重新算一遍」——R1-Zero 在 RL 中 **自发出现** 的行为。

**Aha Moment（顿悟时刻）**  
R1-Zero 训练中出现：模型自己写出类似「Wait, wait. That's an aha moment…」并重新评估思路。说明 RL 能催生复杂策略，而非全靠人工教。

**Pass@1 / Cons@64**  
- **Pass@1**：只生成 1 次，对的概率  
- **Cons@64**（或 majority voting）：生成 64 次，多数表决  
R1-Zero 在 AIME 2024 上 Pass@1 从 15.6% → 71.0%，Cons@64 达 86.7%。

### 2.4 模型架构术语

**Dense Model（稠密模型）**  
每个 token 激活 **全部** 参数。参数多少，算力就多少。

**MoE（Mixture of Experts，混合专家）**  
很多「专家子网络」，每个 token 只路由到 **少数几个专家**。  
- 总参数大（知识库大）  
- 单次计算量小（省钱、快）  
DeepSeek-V3：**671B 总参数，每 token 激活 37B**。

**Expert（专家）**  
MoE 里的一小块 FFN（前馈网络），擅长某类模式。

**Router / Gating（路由/门控）**  
决定当前 token 该找哪几个专家。类似「这道题该问数学专家还是语文专家」。

**负载均衡（Load Balancing）**  
避免所有 token 都涌向同一专家，其他专家闲置。  
DeepSeek-V3 用 **无辅助损失（Auxiliary-Loss-Free）** 策略：给每个专家加偏置项动态调节，减少对模型效果的损害。

**FFN（Feed-Forward Network，前馈网络）**  
Transformer 层里 attention 之后的「思考层」，MoE 主要改的就是 FFN 部分。

**Attention（注意力）**  
模型决定「当前词该多关注哪些别的词」。

**MHA（Multi-Head Attention，多头注意力）**  
从多个角度同时做 attention。GPT 类模型标配。

**MLA（Multi-head Latent Attention，多头潜变量注意力）**  
DeepSeek 自研。把 Key/Value **压缩成更短的向量** 再存进缓存。  
**好处**：长文本推理时 **KV Cache 更小**，省显存、更快。

**KV Cache（键值缓存）**  
生成文本时，已算过的 Key/Value 存起来，避免重复计算。长 CoT 时 KV Cache 很大，MLA 很重要。

**RoPE（Rotary Positional Embedding，旋转位置编码）**  
告诉模型每个词在序列中的位置（第 1 个、第 2 个…）。

**Multi-Token Prediction（MTP，多词元预测）**  
不只预测下一个词，还预测下下个、再下一个…  
训练信号更密，推理时可做 **投机解码（Speculative Decoding）** 加速。

### 2.5 强化学习算法

**Policy（策略）**  
模型生成文本的行为方式。RL 要优化的是 policy。

**PPO（Proximal Policy Optimization）**  
经典 RL 算法，ChatGPT RLHF 常用。需要 **Critic（评论家）** 网络估计「这步好不好」，Critic 常和 Policy 一样大 → **很贵**。

**GRPO（Group Relative Policy Optimization，组相对策略优化）**  
DeepSeek 用的 RL 算法（来自 DeepSeekMath 等工作）。  
**核心思路**：  
1. 同一道题，让模型答 **一组（如 G 个）** 不同答案  
2. 每个答案得奖励 r₁, r₂, …, r_G  
3. **Advantage（优势）** = 该答案比组内平均好多少：  
   `A_i = (r_i - mean(r)) / std(r)`  
4. **不需要单独的 Critic 模型** → 省大量 GPU

**KL 散度惩罚（KL Penalty）**  
防止 RL 后模型偏离原始模型太远，避免「为了高分胡说八道」。公式里有 `β × D_KL(π_θ || π_ref)`。

**Clip（裁剪）**  
PPO/GRPO 里限制单次更新幅度，训练更稳定。

### 2.6 工程与硬件

**GPU（图形处理器）**  
训练大模型的主要算力。DeepSeek-V3 用 **NVIDIA H800**（面向中国市场的降规版 H100）。

**GPU Hour（GPU 小时）**  
1 块 GPU 跑 1 小时 = 1 GPU hour。  
V3 全程约 **278.8 万 H800 GPU 小时**，按 $2/GPU·h 约 **557.6 万美元**。

**FP8（8 位浮点）**  
低精度数值格式。比 FP16/BF16 更省显存、更快，但更难训稳。  
DeepSeek **首次在超大规模模型上验证 FP8 训练可行**。

**BF16 / FP16**  
16 位浮点，常见训练精度。

**MFU（Model FLOPs Utilization，模型算力利用率）**  
GPU 理论算力用了多少比例。越高越「不浪费卡」。

**并行策略**  
大模型放不进单卡，要切分到多卡：  
- **数据并行（DP）**：不同卡看不同 batch  
- **流水线并行（PP）**：不同卡负责不同层  
- **专家并行（EP）**：MoE 的不同专家放不同卡  
- **张量并行（TP）**：单层矩阵切块  

DeepSeek-V3：**16 路 PP + 64 路 EP + ZeRO-1 DP**，且 **不用 TP**（靠内存优化省成本）。

**DualPipe**  
DeepSeek 自研流水线算法：**计算与通信重叠**，减少 MoE 跨节点通信等待（「空泡」）。

**InfiniBand（IB）**  
服务器之间的高速网络，多机训练通信用。

**NVLink / NVSwitch**  
单节点内 GPU 之间的高速互联。

**HAI-LLM**  
DeepSeek 自研训练框架。

### 2.7 数据与评测

**语料 / Corpus**  
训练用的文本集合。官方不公开精确配方，但说明：高质量、多样化、过滤敏感信息与 PII（个人 identifiable 信息）。

**上下文长度（Context Length）**  
一次能读多少 token。V3 预训练后 **两阶段扩展**：先到 **32K**，再到 **128K**。

**MMLU / MATH / AIME / Codeforces**  
常见 benchmark：  
- **MMLU**：多学科知识选择题  
- **MATH / MATH-500**：数学  
- **AIME**：美国数学邀请赛，很难  
- **Codeforces**：编程竞赛平台，用 Elo 评分  

**Helpfulness / Harmlessness**  
- **有用性**：回答是否解决问题  
- **无害性**：是否安全、无偏见、不教坏  

R1 最后一阶段 RL 同时优化这两项。

---

## 三、全景图：DeepSeek 训练到底分几步？

```
第 0 步：准备
  收集与清洗数据 → 设计 V3 架构 (MoE+MLA+MTP) → 搭建 2048×H800 集群

DeepSeek-V3 预训练
  14.8T token 下一词预测 → 上下文 32K → 128K

V3 后训练
  SFT 指令对齐 → RL 人类偏好 → 从 R1 蒸馏推理能力

DeepSeek-R1 四阶段
  Stage1 Cold Start SFT (数千条长CoT)
    → Stage2 推理 RL + GRPO + 规则奖励
    → Stage3 拒绝采样 (~80万条) + SFT
    → Stage4 混合 RL (推理+通用偏好)

可选：蒸馏小模型
  R1 生成数据 → 微调 Qwen/Llama 1.5B~70B
```

**关键关系**：  
- **V3-Base** 是 **R1** 的「原材料」  
- **R1** 训好后，能力 **再灌回 V3**（蒸馏），让通用模型也变强  
- **R1-Zero** 是实验分支：证明 **纯 RL、几乎零 SFT** 也能出推理能力

---

## 四、第 0 步：训练之前要做什么？

### 4.1 数据准备（预训练语料）

DeepSeek 官方算法说明描述：

1. **来源**：公开网页、书籍、代码等（具体比例未公开）  
2. **清洗**：去重、去低质、过滤敏感/个人敏感信息  
3. **目标**：多样、高质量、尽量多语言（中英尤其重要）

### 4.2 选架构：为什么 DeepSeek 用 MoE + MLA？

**问题 1**：要 671B 知识量，Dense 模型训练/推理都太贵。  
**解法**：MoE——「图书馆很大，但每次只开几个书架」。

**问题 2**：R1 会长篇大论，KV Cache 爆炸。  
**解法**：MLA——压缩 KV，128K 上下文才现实。

**问题 3**：预训练效率。  
**解法**：MTP 一次预测多个未来 token，训练信号更密；FP8 降低精度换速度。

### 4.3 集群

- **2048 张 H800**  
- 每节点 8 卡 NVLink  
- 节点间 InfiniBand  
- 自研 **HAI-LLM** + **DualPipe**

---

## 五、DeepSeek-V3 预训练（最烧钱的一步）

### 5.1 在学什么？

**任务**：因果语言建模（Causal LM）——读 `[词1, 词2, …, 词t]`，预测 **词 t+1**。

### 5.2 训练规模（论文数字）

| 项目 | 数值 |
|------|------|
| 语料量 | **14.8T tokens** |
| 预训练 GPU 时 | **266.4 万 H800·h** |
| 每 1T token 耗时 | **18 万 H800·h**（2048 卡约 **3.7 天/T**） |
| 预训练墙钟时间 | **不到 2 个月** |
| 精度 | **FP8 混合精度** |
| 稳定性 | **无不可恢复 loss spike，无 rollback** |

### 5.3 上下文扩展（Context Extension）

1. **第一阶段** → **32K tokens**  
2. **第二阶段** → **128K tokens**  
扩展 GPU 时：**11.9 万 H800·h**

### 5.4 预训练产出

**DeepSeek-V3-Base**：知识、代码、数学底子强，还不会 Chat 格式。

---

## 六、DeepSeek-V3 后训练（把 Base 变成「好用的 Chat」）

预训练后 GPU：**约 5000 H800·h**（论文 Table 1）。

### 6.1 监督微调（SFT）

教模型听指令、安全语气、格式规范。

### 6.2 强化学习（RL）对齐偏好

用奖励模型 + PPO 类流程对齐人类偏好。

### 6.3 从 R1 蒸馏推理能力

R1 的长 CoT、自我验证、反思模式有控制地蒸馏进 V3。

---

## 七、DeepSeek-R1：四阶段流水线（全文重点）

基座：**DeepSeek-V3-Base**

### 7.0 支线：DeepSeek-R1-Zero（纯 RL 实验）

- 起点：V3-Base，**不做 SFT**
- 算法：**GRPO**
- 奖励：**规则型**（Accuracy + Format）
- **不用** 神经网络 Reward Model

**结果**：AIME Pass@1 15.6% → 71.0%；Cons@64 86.7%；自发 reflection 与 Aha moment。

**缺点**：可读性差、中英混杂。

### Stage 1：Cold Start SFT

- **数千条** 长 CoT
- 来源：few-shot、prompt 生成、R1-Zero 输出清洗、人工后处理
- 微调 V3-Base 作为 RL 初始 actor

### Stage 2：Reasoning-oriented RL

- **GRPO + 规则奖励**
- 新增 **语言一致性奖励** 减轻混语
- 训练至推理 benchmark 收敛

### Stage 3：Rejection Sampling + SFT

- 推理数据 **~60 万**：拒绝采样，只留正确轨迹
- 非推理数据 **~20 万**：写作、QA、翻译等
- 合计 **~80 万**，V3-Base 上 **SFT 2 epoch**

### Stage 4：RL for All Scenarios

- 推理题：规则奖励
- 开放题：神经网络 RM + 人类偏好
- Helpfulness 评摘要；Harmlessness 评全文

**产出**：**DeepSeek-R1**

---

## 八、R1 训练成本（与 V3 对比）

| 阶段 | 大致 GPU 成本 | 说明 |
|------|---------------|------|
| V3 预训练 | ~**532 万美元** | 绝对大头 |
| V3 上下文扩展 | ~**24 万美元** | |
| V3 后训练 | ~**1 万美元** | 论文 5K GPU·h |
| R1 核心 RL | ~**100 万美元量级** | 第三方估算 |

---

## 九、蒸馏：DeepSeek-R1-Distill 系列

用 R1 生成推理轨迹，SFT 微调 Qwen2.5 / Llama3（1.5B～70B）。

**结论**：直接蒸馏 R1 > 在小模型上从头做 RL。

---

## 十、时间线

```
2024 及以前 → DeepSeek-V2（验证 MLA + DeepSeekMoE）
2024.12     → DeepSeek-V3 技术报告
2025.01     → DeepSeek-R1 论文 / 开源 + Distill 系列
```

---

## 十一、DeepSeek 与「传统 ChatGPT 路线」对比

| 维度 | 传统（如早期 GPT） | DeepSeek |
|------|-------------------|----------|
| 预训练 | 海量 token + Dense/MoE | 14.8T + MoE 671B/37B active |
| 推理能力 | 大量人工 CoT SFT | **RL 为主**，CoT **涌现** |
| RL 算法 | PPO + Reward Model | 推理：**GRPO + 规则奖励** |
| 架构创新 | 相对标准 | MLA、无辅助损失 MoE、MTP、FP8、DualPipe |
| 开源 | 多闭源 | 权重 + 论文（R1 MIT） |

---

## 十二、官方没完全公开的部分

1. 预训练精确数据配比与全部清洗规则  
2. 各阶段完整超参  
3. V3 SFT 具体条数  
4. 部分 Reward Model 训练细节  

---

## 十三、用一句话串起全流程

1. 在 2048×H800 上，用 MoE+MLA+MTP+FP8，读 14.8T token，训出 DeepSeek-V3-Base  
2. SFT+RL 变成 DeepSeek-V3 Chat  
3. 在 V3-Base 上：Cold Start → GRPO 推理 RL → 80 万拒绝采样 SFT → 混合 RL，得到 DeepSeek-R1  
4. R1 能力蒸馏回 V3，并蒸馏到小模型，形成产品矩阵

---

## 十四、给小白的三条「记忆锚点」

1. **预训练 = 读万卷书**（贵、慢、打底子）  
2. **SFT = 听老师标准答案**（学格式、学听话）  
3. **R1 的 RL = 奥数刷题改错**（规则打分、GRPO 省算力、思考越长分可能越高）

---

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)
- [DeepSeek 模型算法说明](https://cdn.deepseek.com/policies/en-US/model-algorithm-disclosure.html)
- [Epoch AI: What went into training DeepSeek-R1?](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)

---

## 深挖索引（子 agent 已产出）

同目录下 5 份专题深挖文档（术语全解释 + 流程图/伪代码），可按需跳转：

| 文件 | 主题 | 规模 |
|------|------|------|
| [deepseek-grpo-rl-deep-dive-zh.md](/part-07-theory/grpo-rl) | GRPO / PPO / RLVR / 奖励设计 / 训练伪代码 / FAQ | ~618 行 |
| [deepseek-v3-architecture-deep-dive-zh.md](/part-07-theory/v3-architecture) | MoE / MLA / MTP / FP8 / DualPipe / 集群拓扑 | ~583 行 |
| [deepseek-r1-pipeline-deep-dive-zh.md](/part-07-theory/r1-pipeline) | R1-Zero / 四阶段 / Aha moment / o1 对比 / R1-lite 复现 | ~578 行 |
| [deepseek-v3-pretrain-posttrain-deep-dive-zh.md](/part-07-theory/v3-pretrain-posttrain) | 预训练数据 / 14.8T 类比 / YaRN / 成本表 / V3 后训练 | ~548 行 |
| [deepseek-distill-reproduction-deep-dive-zh.md](/part-07-theory/distill-reproduction) | Distill 型号表 / 社区复现 / 消费级部署 / MIT 许可 | ~414 行 |
