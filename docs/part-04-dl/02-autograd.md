# 自动求导

> **前置知识**：张量基础  
> **预计时间**：50 分钟  
> **本章产出**：理解 backward()

PyTorch 自动构建计算图。`y.backward()` 后 `x.grad` 即 ∂y/∂x。

这是神经网络训练的引擎。

## 动手练习

对 y=x³ 在 x=1 求导

## 示例文件

- [`examples/part-04-dl/02-autograd/main.py`](/examples/part-04-dl/02-autograd/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-04-dl/02-autograd/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-04-dl/03-neural-networks)
