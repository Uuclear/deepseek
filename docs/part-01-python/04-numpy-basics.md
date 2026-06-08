# NumPy 入门

> **前置知识**：文件 IO  
> **预计时间**：60 分钟  
> **本章产出**：会用 ndarray 做向量运算

NumPy 是数值计算基础，深度学习张量概念与之类似。

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
a.shape  # (2, 2)
a @ b    # 矩阵乘
a.mean(), a.std()
```

**广播**：不同 shape 数组可自动对齐运算。

## 动手练习

创建一个 3×3 单位矩阵并求逆

## 示例文件

- [`examples/part-01-python/04-numpy-basics/main.py`](/examples/part-01-python/04-numpy-basics/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-01-python/04-numpy-basics/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-01-python/05-pandas-intro)
