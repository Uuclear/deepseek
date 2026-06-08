# 保存与加载 · MNIST

> **前置知识**：过拟合  
> **预计时间**：90 分钟  
> **本章产出**：MNIST test acc > 95%

**里程碑**：运行 [`train_mnist.py`](/examples/part-04-dl/05-training-loop/train_mnist.py) 完整训练。

`torch.save` / `load_state_dict` 持久化权重。

## 动手练习

保存 MNIST 模型并加载推理一张图

## 示例文件

- [`examples/part-04-dl/05-training-loop/train_mnist.py`](/examples/part-04-dl/05-training-loop/train_mnist.py) — MNIST 训练（里程碑）
- [`examples/part-04-dl/07-save-load/main.py`](/examples/part-04-dl/07-save-load/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-04-dl/05-training-loop/train_mnist.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-05-nlp/01-tokenization-embedding)
