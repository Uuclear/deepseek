# 导数与梯度

> **前置知识**：向量基础  
> **预计时间**：60 分钟  
> **本章产出**：理解梯度下降一步

**导数**：函数在某点的变化率。(f(x)=x^2) 在 (x=2) 处导数为 4。

**偏导数**：多变量时对某一变量的导数。
**梯度**：所有偏导组成的向量，指向上升最快方向。

**梯度下降**：`x = x - lr * grad`，沿负梯度走以减小损失。

## 动手练习

对 f(x,y)=x²+y² 手写 5 步梯度下降

## 示例文件

- [`examples/part-02-math/02-gradients/main.py`](/examples/part-02-math/02-gradients/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-02-math/02-gradients/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-02-math/03-probability)
