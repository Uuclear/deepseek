# 模型量化与导出（GGUF / ONNX 直觉）

<LearningMeta
  prereq="<a href='/part-08-llm-build/07-eval-perplexity'>Part 8-07 评估</a>、<a href='/part-08-llm-build/08-scale-to-opensource'>Part 8-08 开源生态</a>"
  time="75 分钟"
  output="理解 FP16/INT4/GGUF/ONNX 适用场景，能列出导出检查清单"
/>

训练得到的是 **浮点权重**；部署到手机、边缘设备或高并发 API 时，常需要 **量化** 与 **格式转换**。

## 精度与体积

| 格式 | 每参数约 | 7B 模型体积（粗算） | 典型场景 |
|------|----------|---------------------|----------|
| FP32 | 4 B | ~28 GB | 训练 |
| FP16/BF16 | 2 B | ~14 GB | 训练/推理 |
| INT8 | 1 B | ~7 GB | 推理加速 |
| INT4 / Q4 | 0.5 B | ~3.5 GB | 本地 Ollama、llama.cpp |

量化 = 用更少 bit 近似表示权重，**略损精度，大幅减体积、提吞吐**。

## GGUF 直觉

**GGUF**（原 GGML 演进）是 llama.cpp / Ollama 生态常用格式：

- 单文件打包权重 + 元数据（架构、词表、量化类型）
- 支持 Q4_K_M、Q8_0 等多种量化方案
- CPU/GPU 混合推理友好

```powershell
# 概念流程（需安装对应工具链）
# 1. HuggingFace 模型 → 转换脚本 → GGUF
# 2. ollama create -f Modelfile  # 引用 GGUF 或 HF 路径
```

个人学习：**会用 Ollama 拉取量化模型即可**；自训练 mini_gpt 体量太小，不必转 GGUF，但需理解工业流程。

## ONNX 直觉

**ONNX** 是跨框架的中间表示，便于：

- PyTorch → ONNX → TensorRT / ONNX Runtime
- 固定计算图，利于编译优化
- 嵌入式、移动端部署

适用：**推理服务标准化**；不适合直接承载 LLM 全部训练动态图。

```python
# 示意：小模型可导出 ONNX（大模型需专用工具）
import torch
dummy = torch.randint(0, 100, (1, 64))
torch.onnx.export(model, dummy, "mini.onnx", opset_version=17)
```

## 导出检查清单

- [ ] 词表与 `config.json` 一并保存
- [ ] 记录 `max_seq_len`、RoPE 参数
- [ ] 量化后跑 [Golden Set](/data/eval_questions.jsonl) 对比答案率
- [ ] 文档化依赖版本（CUDA、cuDNN、推理引擎）

::: warning 精度回退
激进量化（Q2/Q3）可能明显掉点；生产环境用 Q4_K_M 或 Q8 并做 A/B 评测。
:::

## 与 Part 6 / Part 10 的衔接

- 本地部署：[Part 6 部署](/part-06-practice/00-deployment)（Ollama 即消费量化模型）
- 高吞吐服务：[Part 10 vLLM](/part-10-advanced/04-vllm-deployment)
- 蒸馏后小模型更适合量化部署：[Part 10-03 蒸馏](/part-10-advanced/03-distillation-practice)

<ExampleBox
  title="评测数据"
  path="data/eval_questions.jsonl"
  download="/data/eval_questions.jsonl"
  command="python examples/part-10-advanced/eval_runner/eval_llm.py"
/>

---

上一章：[08-09 分布式训练](/part-08-llm-build/09-distributed-training) · 下一章：[Part 9 Agent 概述](/part-09-agents/01-agent-overview)
