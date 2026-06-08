# RL 与进阶训练路线图

> **前置**：完成 [guide-02 SFT](/part-06-practice/02-finetuning)、[guide-03 数据](/part-06-practice/03-data-evaluation)。  
> **目标**：理解 **何时需要 RL**、个人能复现到哪一步、用什么工具。  
> **理论对照**：[deepseek-grpo-rl-deep-dive-zh.md](/part-07-theory/grpo-rl)、[deepseek-r1-pipeline-deep-dive-zh.md](/part-07-theory/r1-pipeline)

---

## 1. SFT 不够时，才考虑 RL

| 方法 | 解决什么 | 不解决什么 |
|------|----------|------------|
| **Prompt** | 格式、单次任务 | 稳定领域能力 |
| **RAG** | 私有知识检索 | 推理习惯、风格 |
| **SFT** | 模仿示范、领域格式 | 探索更优推理路径 |
| **RL（RLVR）** | 可验证任务上 **推极限**、涌现长 CoT | 开放题主观偏好（需 RM） |

DeepSeek 路径：**SFT 打底 → RL 激发推理 → 再 SFT 补通用 → 混合 RL 对齐**（四阶段）。

个人 realistic 路径：**Distill（SFT）→（可选）GRPO on 7B**。

---

## 2. RL 核心术语（实操版）

**Policy（策略）**  
当前模型；RL 更新它的生成概率。

**Rollout（ rollout ）**  
用当前模型对一批 prompt **采样生成** 完整回答。

**Reward（奖励）**  
对每个回答打分。数学：**对=1 错=0**；格式：**有思考标签 +0.1**。

**Advantage（优势）**  
GRPO 里：同一题多个回答，比组内平均好多少 → 加强/减弱该回答 token 的概率。

**Reference Model（参考模型）**  
RL 前的 checkpoint；KL 惩罚防止偏离太远。

**GRPO**  
DeepSeek R1 用的算法；比 PPO **少一个 Critic 大模型**，省 GPU。详见 [GRPO 深挖](/part-07-theory/grpo-rl)。

---

## 3. 个人可复现的三条路线

### 路线 1：TinyZero / 玩具 RL（理解涌现，<$50）

| 项目 | 说明 |
|------|------|
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | Countdown 等小游戏，2 卡，PPO/GRPO |
| 目标 | 看到 **Aha moment**、自我验证，**不是** 刷 AIME |
| 适合 | 理解 RL loop，零 RL 论文阅读补充 |

### 路线 2：Distill + SFT（主力，单卡～多卡）

| 步骤 | 说明 |
|------|------|
| Teacher | `deepseek-reasoner` API |
| 数据 | 5k～50k，拒绝采样 + 数学验证 |
| 训练 | LLaMA-Factory LoRA on Distill-7B |
| 期望 | 接近社区 Open-R1 水平，低于官方 800k Distill |

### 路线 3：Distill + GRPO（进阶，多卡 / 云租）

| 步骤 | 说明 |
|------|------|
| 起点 | 官方 `DeepSeek-R1-Distill-Qwen-7B` 或你的 SFT checkpoint |
| 框架 | [veRL](https://github.com/volcengine/verl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL GRPOTrainer](https://huggingface.co/docs/trl) |
| 数据 | 数学/代码 **可验证** prompt |
| 奖励 | 规则（答案对错 + 格式） |
| 参考 | DeepScaleR、Skywork-OR1 在 Distill 上继续 GRPO |

**硬件粗估（7B GRPO）**  
- 组大小 G=8～16，每 prompt 生 2k～4k token  
- **至少 1×A100 40GB** 或 **2×4090** 较稳；消费级单卡可试 TRL + 小 batch

---

## 4. GRPO 训练 loop（与代码框架对应）

```text
for step in range(num_steps):
    batch = sample_prompts()           # 数学题、代码题
    for q in batch:
        outputs = [model.generate(q) for _ in range(G)]  # 一组 G 个
        rewards = [rule_reward(q, o) for o in outputs]   # 对错、格式
        advantages = group_normalize(rewards)            # GRPO
        loss = grpo_loss(outputs, advantages, ref_model) # + KL
    optimizer.step()
```

**规则奖励示例（数学）**：

```python
def math_reward(question: str, response: str, ground_truth: str) -> float:
    pred = extract_boxed(response)
    if pred is None:
        return 0.0
    ok = verify_equal(pred, ground_truth)  # sympy / math-verify
    fmt = 0.1 if "" in response else 0.0  # 格式分，以实际模板为准
    return float(ok) + fmt
```

**R1 论文要点（复现时别踩坑）**  
- 推理 RL 用 **规则奖**，少用 neural RM（防 reward hacking）  
- GRPO 的 **clip ε 很大（~10）**，长 CoT 需要大 clip  
- KL 在 **loss 里** 而非逐步 reward 里（利于回答变长）  
- Stage 4 才加 Helpfulness RM，且 **只评 summary**

---

## 5. 框架选型

| 框架 | 适合 | 备注 |
|------|------|------|
| **TRL `GRPOTrainer`** | HF 生态、7B 试验 | 文档全，入门快 |
| **veRL** | 中大规模 RL | 字节开源，R1 复现讨论多 |
| **OpenRLHF** | PPO/RM 全流程 | 偏 RLHF，也可 RLVR |
| **LLaMA-Factory** | 仅 SFT | RL 需另接 |

**TRL 最小概念验证**（需自行补 reward 函数与数据集）：

```bash
pip install trl transformers accelerate
# 参考 TRL 文档 GRPO 章节 + GSM8K 子集
```

---

## 6. 从 DeepSeek 四阶段到个人裁剪版

| 官方阶段 | 个人裁剪 |
|----------|----------|
| R1-Zero 纯 RL | TinyZero 或 7B + 小数学集 GRPO |
| Cold Start 数千条 | 500 条人工/Teacher CoT SFT |
| 推理 RL 至收敛 | 云 GPU 跑 1k～5k step（看 val） |
| 600k+200k SFT | 5k～20k 混合 SFT |
| 混合 RL | 可选；Helpfulness 用 GPT/DeepSeek 当 Judge |

**不要一步到位复现 R1**；按 **路线 2 → 路线 3** 递进。

---

## 7. 预训练：个人边界

| 项目 | 是否 realistic |
|------|----------------|
| 从头预训练 7B | 机构级（百万美元级） |
| 继续预训练（CPT）领域语料 | 小团队可试 **1B～7B + 小语料** |
| 14.8T DeepSeek-V3 预训练 | **否** |

若要做 CPT：参考 [继续预训练指南](https://huggingface.co/learn/nlp-course) 小章节，语料 **100M～1B token** 级起步。

---

## 8. 环境与工程清单（进阶训练）

```bash
# 常见栈
Python 3.10+
PyTorch 2.x + CUDA 12.x
flash-attn（可选，长上下文）
deepspeed / FSDP（多卡）
wandb 实验跟踪
```

**仓库隔离**：训练代码与数据分目录；**checkpoint** 定期存 S3/磁盘。

---

## 9. 失败模式与排查

| 现象 | 可能原因 |
|------|----------|
| reward 涨、评测跌 | reward hacking；检查规则漏洞 |
| 输出乱码/混语 | 缺 language consistency 奖励或 Cold Start |
| KL 爆炸 | β 太小或 lr 太大 |
| 长度不增长 | KL 在 reward 里（PPO 式）或 clip 太小 |
| OOM | 减 G、减 max_new_tokens、QLoRA policy |

---

## 10. 自检

- [ ] 能口头解释 SFT vs RLVR 分工
- [ ] 跑过至少一次 SFT（guide-02）
- [ ] （可选）跑 TinyZero 或 TRL GRPO demo
- [ ] 读过 [GRPO 深挖](/part-07-theory/grpo-rl) 的 FAQ

总路线图见 [guide-05](/part-06-practice/05-roadmap)。
