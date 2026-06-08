# 进阶专题速查（原创）

## MoE 与路由

Mixture of Experts 通过路由层为每个 token 选择 top-k 专家 FFN，总参数量大但激活参数可控。负载均衡损失避免专家闲置。

## GRPO 强化学习

Group Relative Policy Optimization 对同一 prompt 的多条回答组内打分，优于平均的增大概率，并加 KL 约束防止偏离 SFT 策略过远。

## vLLM 与 PagedAttention

vLLM 用分页管理 KV Cache，配合连续批处理提升推理吞吐，适合高并发 Agent 后端。

## 多模态 Vision + LLM

图像经 ViT 编码为视觉 token，经 projector 对齐语言嵌入后与文本 token 一并送入 Transformer。

## LoRA 低秩适配

LoRA 在注意力层旁路注入低秩矩阵，只训练少量参数即可适配下游任务，是 SFT 的常用高效方案。

## 蒸馏与温度

Teacher-Student 蒸馏用温度 T>1 软化 softmax 分布，让 Student 学习 Teacher 的软标签暗知识，常与硬标签 CE 混合。
