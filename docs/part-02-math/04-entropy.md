# 信息论与熵

> **前置知识**：概率基础  
> **预计时间**：45 分钟  
> **本章产出**：会算二元熵

**熵 H**：衡量不确定性。公平硬币 H=1 bit。

公式：(H = -\sum p \log_2 p)

越确定（如 99%/1%）熵越低。交叉熵常作分类损失。

## 动手练习

计算四分类均匀分布的熵

## 示例文件

- [`examples/part-02-math/04-entropy/main.py`](/examples/part-02-math/04-entropy/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-02-math/04-entropy/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-02-math/05-loss-functions)
