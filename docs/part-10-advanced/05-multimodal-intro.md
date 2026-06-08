# 多模态入门（Vision + LLM 概念）

<LearningMeta
  prereq="<a href='/part-05-nlp/04-transformer'>Part 5 Transformer</a>、<a href='/part-09-agents/01-agent-overview'>Part 9 Agent</a>"
  time="60 分钟"
  output="理解图像编码 + 投影到 LLM 词嵌入空间的基本范式"
/>

**多模态大模型** 让 LLM 不仅读文本，还能「看」图像、听音频（本章聚焦 **Vision + LLM**）。

## 典型架构

```text
图像 → Vision Encoder（ViT / CLIP）→ 视觉 token 序列
                                      ↓
                              Projector（MLP）
                                      ↓
文本 token ──────────────→  LLM Transformer  →  文本回答
```

视觉 token 与文本 token **拼接** 后送入同一 Transformer，自回归生成说明或答案。

## 训练阶段（直觉）

1. **对齐**：大量图文对，训练 projector 对齐视觉与语言空间
2. **指令微调**：多模态对话数据，学会遵循指令
3. **（可选）RLHF/GRPO**：提升有用性与安全性

## Agent 场景

- 截图理解 + 操作 GUI（计算机使用 Agent）
- 图表、票据 OCR 问答
- 与 [RAG](/part-09-agents/05-rag-agent) 结合：检索文档 + 理解插图

::: info 本教程边界
仓库不提供大规模视觉训练；建议用开源 VLM API 或本地 `llava` 类模型做实验。原理与文本 LLM 共享 Transformer 直觉。
:::

## 动手练习

1. 列举 3 个多模态 prompt（图表解读、菜单翻译、缺陷检测）
2. 说明为何视觉 token 数影响显存（与文本 context 同类问题）
3. 画一张「RAG + Vision」Agent 数据流

---

上一章：[10-04 vLLM 部署](/part-10-advanced/04-vllm-deployment)
