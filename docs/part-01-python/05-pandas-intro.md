# pandas 入门

> **前置知识**：NumPy  
> **预计时间**：60 分钟  
> **本章产出**：会读 CSV、做简单统计

pandas `DataFrame` 类似 Excel 表：

```python
import pandas as pd
df = pd.read_csv("data/iris.csv")
df.head()
df["species"].value_counts()
df.select_dtypes("number").mean()
```

**筛选**：`df[df["sepal_length"] > 5.5]`

## 动手练习

读取 titanic_sample.csv，计算各舱位生存率

## 示例文件

- [`examples/part-01-python/05-pandas-intro/main.py`](/examples/part-01-python/05-pandas-intro/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-01-python/05-pandas-intro/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-02-math/01-vectors-matrices)
