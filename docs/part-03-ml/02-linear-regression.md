# 线性回归

> **前置知识**：监督学习概念  
> **预计时间**：60 分钟  
> **本章产出**：在 Iris 上拟合线性关系

模型：`ŷ = w·x + b`，最小化 MSE。

sklearn：`LinearRegression().fit(X, y)`

看 **系数**、**R²**、**RMSE** 判断拟合好坏。

## 动手练习

用两个特征预测 sepal_length

## 示例文件

- [`examples/part-03-ml/01-linear-regression/train.py`](/examples/part-03-ml/01-linear-regression/train.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-03-ml/01-linear-regression/train.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-03-ml/03-logistic-regression)
