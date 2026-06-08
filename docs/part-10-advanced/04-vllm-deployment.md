# vLLM 推理部署

<LearningMeta
  prereq="<a href='/part-06-practice/00-deployment'>Part 6 部署</a>、<a href='/part-08-llm-build/10-quantization-export'>Part 8 量化导出</a>"
  time="60 分钟（概念 + 可选实操）"
  output="理解 PagedAttention、连续批处理，能列出 vLLM 部署检查项"
/>

**vLLM** 是面向生产的高吞吐 LLM 推理引擎，核心优化 **KV Cache 内存** 与 **动态批处理**。

## 痛点

自回归生成时，KV Cache 随序列变长而膨胀；传统服务 **预留最大长度** → 显存浪费严重。

## PagedAttention 直觉

借鉴操作系统虚拟内存 **分页**：

- KV Cache 切成固定大小 block
- 序列逻辑连续，物理 block 可不连续
- 按需分配/回收 block → 显存利用率大幅提升

## 连续批处理（Continuous Batching）

新请求可随时插入 batch；完成的序列立即腾出算力 — 不像静态 batch 必须等同批最长序列结束。

## 快速体验（Linux / WSL2 推荐）

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000
```

Windows 原生支持有限，教程以 **概念理解** 为主；本地体验可继续用 [Ollama](/part-06-practice/00-deployment)。

::: tip 与 Agent
高并发 Agent 平台常在 vLLM 后挂 OpenAI 兼容 API，多个 Agent 共享同一推理池。
:::

## 部署检查清单

- [ ] GPU 显存与 `max_model_len` 匹配
- [ ] 量化权重格式（AWQ/GPTQ）与 vLLM 版本兼容
- [ ] 限流、超时、日志与 [Golden Set](/data/eval_questions.jsonl) 冒烟
- [ ] 对比延迟：vLLM vs 单进程 HF `generate`

---

上一章：[10-03 蒸馏实战](/part-10-advanced/03-distillation-practice) · 下一章：[10-05 多模态入门](/part-10-advanced/05-multimodal-intro)
