# 蒸馏：Teacher-Student 实战

<LearningMeta
  prereq="<a href='/part-07-theory/distill-reproduction'>Part 7 蒸馏与复现</a>、<a href='/part-08-llm-build/02-mini-gpt-arch'>Part 8 mini GPT</a>"
  time="90 分钟"
  output="跑通简化蒸馏 demo，理解软标签与温度"
/>

**知识蒸馏**：大模型（Teacher）指导小模型（Student）学习，使小模型在有限算力下逼近大模型行为。

## 核心公式直觉

Student 损失常含两项：

```text
L = α · KL(softmax(z_t/T), softmax(z_s/T)) + (1-α) · CE(y, z_s)
```

- `z_t` / `z_s`：Teacher / Student logits
- `T`：温度，T>1 使分布更平滑，传递「暗知识」
- `α`：平衡蒸馏与硬标签

## 本仓库 demo

不依赖大模型 API，用 **随机 Teacher** 演示 KL 蒸馏训练循环（教学用）：

```powershell
python examples/part-10-advanced/distill_demo/distill_train.py --epochs 5
```

<ExampleBox
  title="蒸馏 Demo"
  path="examples/part-10-advanced/distill_demo/"
  command="python examples/part-10-advanced/distill_demo/distill_train.py --epochs 5"
/>

## 与 DeepSeek 蒸馏

DeepSeek-R1 系列包含 **长链推理蒸馏** 到较小模型 — 数据是高质量 reasoning trace，而不只是 logits。详见 [Part 7](/part-07-theory/distill-reproduction)。

## 实践建议

1. 先 SFT Student 到可用基线
2. 用 Teacher 对同一 prompt 生成多条回答，筛高质量进数据集（**序列级蒸馏**）
3. 可选 logits 蒸馏（需 Teacher 开放 logits，成本高）
4. 蒸馏后做 [量化](/part-08-llm-build/10-quantization-export) 部署

---

上一章：[10-02 MoE 架构](/part-10-advanced/02-moe-deepseek-arch) · 下一章：[10-04 vLLM 部署](/part-10-advanced/04-vllm-deployment)
