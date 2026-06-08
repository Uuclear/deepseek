# SFT / LoRA 微调入门（以 DeepSeek-R1-Distill 为例）

> **前置**： [guide-00 部署](/part-06-practice/00-deployment)、[guide-01 应用开发](/part-06-practice/01-inference)。  
> **目标**：用 **自己的数据** 改变模型行为，而不从头预训练。

---

## 1. 先搞清：你在做什么

**预训练**  
读万亿 token，学语言与知识。**个人不做。**

**SFT（Supervised Fine-Tuning，监督微调）**  
用「输入 → 标准输出」样本继续训练，学 **格式、风格、领域、任务**。

**LoRA（Low-Rank Adaptation，低秩适配）**  
只训练少量附加参数（常 <1%），原模型权重冻结或合并。  
**个人/单卡最常用**，显存友好。

**全参微调（Full Fine-tuning）**  
更新全部权重。7B 需 ~60GB+ 显存（FP16），一般机构才做。

**DeepSeek 官方 Distill 路线**  
R1 生成 80 万条 → 对 Qwen/Llama **SFT**。你复现时用 **几百～几万条** 也能看到效果。

---

## 2. 硬件与模型选择

| 方案 | 基座模型 | 显存（LoRA） | 工具 |
|------|----------|--------------|------|
| 入门 | `DeepSeek-R1-Distill-Qwen-1.5B` | 8GB | LLaMA-Factory / Unsloth |
| 推荐 | `DeepSeek-R1-Distill-Qwen-7B` | 16～24GB | LLaMA-Factory |
| 进阶 | `DeepSeek-R1-Distill-Qwen-14B` | 24～48GB | QLoRA 4bit |

权重：[HuggingFace deepseek-ai](https://huggingface.co/deepseek-ai)

---

## 3. 数据格式（Alpaca / ShareGPT）

### 3.1 Alpaca 格式（单轮为主）

```json
[
  {
    "instruction": "将下面句子翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is nice today."
  }
]
```

### 3.2 ShareGPT 格式（多轮）

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么可以帮你？"}
    ]
  }
]
```

### 3.3 推理 / CoT 格式（模仿 R1）

```json
{
  "instruction": "解方程 2x + 3 = 11",
  "input": "",
  "output": "\n2x = 11 - 3 = 8, x = 4\n\n\n最终答案：x = 4"
}
```

标签名以你用的训练框架模板为准（DeepSeek-R1 官方用 `redacted_reasoning` 等 special token，Distill 模型 HF 卡片有说明）。

**数据量参考**：

| 场景 | 条数 | 说明 |
|------|------|------|
| 风格/格式 | 100～500 | 很快过拟合，注意验证集 |
| 领域 QA | 1k～10k | 配合 RAG 更好 |
| 推理蒸馏 | 5k～50k | 社区 Open-R1 量级 |
| 官方 R1 蒸馏 | ~800k | 个人难完全复现 |

---

## 4. 路线 A：LLaMA-Factory（推荐，GUI + CLI）

### 4.1 环境（WSL2 / Linux 推荐；Windows 可用 Docker）

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

需：**Python 3.10+**、**CUDA**、**PyTorch**（与 CUDA 版本匹配）。

### 4.2 准备数据

将 JSON 放到 `data/`，并在 `data/dataset_info.json` 注册：

```json
{
  "my_sft": {
    "file_name": "my_sft.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 4.3 LoRA 训练配置示例

创建 `examples/train_lora/deepseek_distill_7b.yaml`：

```yaml
model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: my_sft
template: deepseekr1
cutoff_len: 4096
max_samples: 1000
overwrite_cache: true

output_dir: saves/deepseek-7b-lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

logging_steps: 10
save_steps: 500
plot_loss: true
```

```bash
llamafactory-cli train examples/train_lora/deepseek_distill_7b.yaml
```

### 4.4 Web UI

```bash
llamafactory-cli webui
```

浏览器里选模型、数据集、LoRA 参数，适合第一次试跑。

### 4.5 合并 LoRA 并推理

```bash
llamafactory-cli export examples/merge_lora/deepseek_7b.yaml
```

或在 vLLM / Ollama 加载合并后的完整权重。

---

## 5. 路线 B：Unsloth（单卡极速 LoRA）

```bash
pip install unsloth
```

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# 接 HuggingFace Trainer + 你的 Dataset
```

适合：**快速实验**、显存紧张（4bit QLoRA）。

---

## 6. 路线 C：用 DeepSeek API 造数据再 SFT（蒸馏入门）

官方 800k 未公开；个人常用 **Teacher = deepseek-reasoner**：

```python
# 伪代码：批量生成 SFT 样本
prompts = open("prompts.txt").read().splitlines()
rows = []
for p in prompts:
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": p}],
    )
    rows.append({"instruction": p, "input": "", "output": resp.choices[0].message.content})
# 保存 JSON → 人工抽检 5% → 再 SFT
```

**数学/代码** 建议加 **自动验证**（SymPy、单元测试）再入库，见 [guide-03](/part-06-practice/03-data-evaluation)。

---

## 7. 超参速查（7B LoRA 默认起点）

| 参数 | 建议 | 说明 |
|------|------|------|
| learning_rate | 1e-4 ~ 5e-5 | 太大易遗忘 |
| lora_rank | 8～64 | 越大越强，易过拟合 |
| epochs | 2～5 | 看验证 loss |
| batch（有效） | 16～128 | 小 batch + 梯度累积 |
| cutoff_len | 2048～8192 | 推理数据要长 |
| warmup | 3%～10% | 稳定初期 |

**过拟合信号**：训练 loss 继续降，验证 loss 升；对 held-out 问题复读训练答案。

---

## 8. 训练后评估

1. **Held-out 测试集**（从未参与训练的问题）
2. **同一 prompt 对比** 微调前后输出
3. **领域 benchmark**（若做数学：GSM8K 子集抽 100 题）
4. **遗忘检查**：问通用知识是否仍正常

不要只看 training loss。

---

## 9. 许可注意

| 模型 | 许可 | 商用 |
|------|------|------|
| DeepSeek-R1 / Distill（MIT） | 宽松 | 一般可，读 LICENSE |
| Qwen2.5 基座 | Apache 2.0 | 可 |
| Llama 3 | Meta License | 有营收门槛等限制 |

微调后的 LoRA 适配器通常可商用，但 **基座许可** 仍约束你。

---

## 10. 常见问题

**Q：微调后模型变笨？**  
A：学习率过大 / epoch 过多 / 数据太少导致 **灾难性遗忘**；减 lr、加通用数据混合、少训几轮。

**Q：7B 训不动？**  
A：QLoRA 4bit、减 `cutoff_len`、减 batch、用 gradient checkpointing。

**Q：和 R1 全量训练差在哪？**  
A：你做的是 **SFT/LoRA**；R1 还有 **GRPO 强化学习**（见 [guide-04](/part-06-practice/04-rl-roadmap)）。

---

## 11. 自检与下一步

- [ ] 准备 ≥100 条 Alpaca 格式数据
- [ ] 用 LLaMA-Factory 或 Unsloth 跑通 1 次 LoRA
- [ ] 合并或加载 adapter 做推理对比
- [ ] 记录 train/val loss 与 10 条人工对比

下一步：[guide-03 数据与评测](/part-06-practice/03-data-evaluation)
