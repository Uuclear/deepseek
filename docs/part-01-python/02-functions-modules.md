# 函数与模块

> **前置知识**：变量与控制流  
> **预计时间**：45 分钟  
> **本章产出**：会写函数、拆分模块

**函数**封装重复逻辑：

```python
def normalize(values):
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values]
```

**模块**：`utils.py` 放工具函数，`main.py` 里 `from utils import normalize`

**类型提示**（可选）：`def f(x: float) -> float:`

## 动手练习

把 normalize 改成 min-max 到 [-1, 1]

## 示例文件

- [`examples/part-01-python/02-functions/main.py`](/examples/part-01-python/02-functions/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-01-python/02-functions/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-01-python/03-file-io)
