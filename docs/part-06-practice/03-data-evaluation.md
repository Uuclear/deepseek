# 数据构造与评测入门

> **前置**：[guide-02 微调](/part-06-practice/02-finetuning) 了解 SFT 格式。  
> **目标**：会 **造数据、洗数据、评数据**——微调与 RL 都依赖数据质量。

---

## 1. 数据在 AI 开发里的位置

```text
部署（会用）→ 应用（调 prompt）→ 造数据 → 微调 → 评测 → （可选）RL
                     ↑__________________|
                          迭代闭环
```

**Garbage in, garbage out**：800 条高质量 > 8 万条垃圾。

---

## 2. 术语

**Prompt**  
给模型的输入（问题/指令）。

**Completion / Response**  
模型输出；SFT 里常叫 `output` / `response`。

**Ground Truth（标准答案）**  
已知正确答案，用于数学、分类、代码测试。

**Golden Set / Eval Set**  
固定评测集，**永不参与训练**，只用来衡量版本好坏。

**Train / Val / Test 划分**  
常见 80/10/10 或 90/5/5；**同一问题不能泄漏**到训练和测试。

**Data Leakage（数据泄漏）**  
评测题出现在训练里 → 分数虚高。

**Rejection Sampling（拒绝采样）**  
多次生成，只保留通过验证的样本（DeepSeek R1 Stage 3 核心）。

---

## 3. 数据来源（个人/小团队）

| 来源 | 做法 | 成本 |
|------|------|------|
| 人工编写 | 领域专家写 Q&A | 高质低量 |
| 业务日志 | 客服/搜索日志脱敏 | 需清洗 |
| 公开数据集 | GSM8K、Alpaca、ShareGPT 子集 | 快，注意许可 |
| Teacher 蒸馏 | `deepseek-reasoner` 批量生成 | API 费 |
| 数据增强 | 同题改写法、换语言 | 扩量 |

**DeepSeek 官方 R1 数据构成（参考）**  
~600k 推理 + ~200k 非推理；推理经 **规则验证 + 拒绝采样**；非推理来自 V3 管线。

---

## 4. 构造 SFT 数据：逐步流水线

### Step 1：定义任务与格式

写一份 **数据规格**（1 页纸）：

```markdown
- 任务：公司内部 IT 问答
- 语言：简体中文
- 格式：Alpaca instruction/input/output
- 长度：output < 512 tokens
- 禁止：编造工单号；不知道时说「请联系 IT 台」
```

### Step 2：收集 Prompt

从真实用户问题、FAQ、工单标题抽取；去重（MinHash / 简单字符串）。

### Step 3：生成 Response

**人工** 或 **Teacher API**：

```python
SYSTEM = "你是 IT 助手。仅根据提供的知识回答；无依据则说不知道。"

def generate_one(question: str, kb_snippet: str) -> str:
    user = f"参考资料：\n{kb_snippet}\n\n问题：{question}"
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content
```

### Step 4：质量过滤

| 检查 | 方法 |
|------|------|
| 长度 | 过短/过长丢弃 |
| 重复 | 与已有 output 相似度 >0.9 丢弃 |
| 格式 | 必须含必要字段 |
| 安全 | 关键词黑名单 |
| 事实 | RAG 场景：答案是否能在资料中找到 |
| 数学 | SymPy 验算 |
| 代码 | 沙箱 `pytest` / 运行样例 |

**数学验证示例**：

```python
import sympy as sp

def check_math_answer(question: str, final_answer: str) -> bool:
    # 简化：从 output 提取 \boxed{} 或最后一行数字，与 sympy 解对比
    # 生产环境用 math-verify 等库
    ...
```

### Step 5：人工抽检

随机 **5%～10%** 人工标 `pass/fail`；fail 的找模式改 prompt 或过滤规则。

### Step 6：划分与版本化

```text
data/
  v1/
    train.json
    val.json
    test.json
    README.md   # 来源、过滤规则、条数、日期
```

用 **DVC / git tag** 标记数据版本，便于复现实验。

---

## 5. 推理 / CoT 数据（模仿 R1 蒸馏）

**结构**：

```json
{
  "instruction": "证明 sqrt(2) 是无理数",
  "output": "\n...长推理...\n\n\n因此 sqrt(2) 是无理数。"
}
```

**Teacher 调用**（`deepseek-reasoner`）：

- `temperature=0.6`, `top_p=0.95`（对齐论文评测采样）
- 保存 **完整** 思考 + 答案

**过滤**（对齐 DeepSeek Stage 3）：

- 去掉混语严重、无 `\boxed{}`（数学题）、代码块混乱
- 数学：答案可验证才保留
- 每 prompt 采样 N 次，**只留正确**

社区规模参考：Open-R1 ~220k，OpenThoughts ~114k，LIMO 仅 **817** 条也能涨分（质量 > 数量）。

---

## 6. 评测体系

### 6.1 三层评测

| 层级 | 测什么 | 例子 |
|------|--------|------|
| **自动** | 可规则判定 | 准确率、BLEU、代码 pass@1 |
| **模型评模型** | LLM-as-Judge | 用强模型打 1～5 分 |
| **人工** | 主观质量 | 可读性、有用性 |

### 6.2 常用 Benchmark（按需选）

| Benchmark | 领域 | 个人复现难度 |
|-----------|------|--------------|
| GSM8K | 小学数学 | 低 |
| MATH-500 | 竞赛数学 | 中 |
| HumanEval | 代码 | 低 |
| MMLU 子集 | 知识 | 中 |
| 自建 Golden Set | 你的业务 | **最该做** |

**原则**：先做 **50～200 条自建 Golden Set**，再跑公开榜。

### 6.3 自动评测脚本骨架

```python
import json

def eval_gsm8k(model_fn, path="test.jsonl"):
    ok = total = 0
    for line in open(path):
        row = json.loads(line)
        pred = model_fn(row["question"])
        truth = row["answer"]
        if extract_number(pred) == extract_number(truth):
            ok += 1
        total += 1
    return ok / total

def extract_number(text: str):
    import re
    m = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None
```

### 6.4 Pass@1 vs Pass@k

- **Pass@1**：每题生成 1 次  
- **Pass@k**：生成 k 次，**任一** 对即算对（训练 RL 时常用 k=16）

报告分数时 **注明 temperature、prompt、模型版本**。

### 6.5 A/B 对比

同一测试集上：

```text
Base 模型分：72%
+ LoRA v1：78%
+ LoRA v2（加数据）：81%
```

避免只报训练 loss。

---

## 7. 数据版本与实验记录

**实验表**（ spreadsheet 或 markdown）：

| run_id | 数据版本 | 基座 | 方法 | GSM8K | 备注 |
|--------|----------|------|------|-------|------|
| 001 | v1 train 1k | distill-7b | LoRA r=8 | 0.62 | baseline |
| 002 | v2 +math 5k | distill-7b | LoRA r=16 | 0.71 | +拒绝采样 |

---

## 8. 工具推荐

| 工具 | 用途 |
|------|------|
| [Label Studio](https://labelstud.io/) | 人工标注 |
| [HuggingFace datasets](https://huggingface.co/datasets) | 托管与加载 |
| [sympy](https://www.sympy.org/) / [math-verify](https://github.com/huggingface/math-verify) | 数学验证 |
| [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准 benchmark |
| [wandb](https://wandb.ai/) | 实验跟踪 |

---

## 9. 自检

- [ ] 有 train/val/test 且 test 从未进训练
- [ ] 写清数据规格与过滤规则
- [ ] Teacher 生成 + 至少一种自动验证
- [ ] 50+ 条 Golden Set + 一次自动评测脚本
- [ ] 实验 run 与数据版本可对应

下一步：[guide-04 RL 与训练路线图](/part-06-practice/04-rl-roadmap)
