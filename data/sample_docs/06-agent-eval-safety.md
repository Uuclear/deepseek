# Agent 评测与安全（教程片段）

## 幻觉

Agent 可能编造未出现在检索结果中的事实。缓解方法包括：强制引用来源、低置信度拒答、Golden Set 回归测试。

## 工具滥用

应使用工具白名单、最大步数限制与沙箱执行。切勿在生产环境直接执行模型生成的 shell 命令。

## Golden Set

评测集应覆盖：任务完成率、工具正确率、groundedness 与安全性。本仓库 eval_questions.jsonl 提供关键词冒烟用例。

## 沙箱

代码类 Agent 应在隔离 subprocess 中运行，设置超时并禁止外网访问，仅返回 stdout 给 LLM 继续推理。
