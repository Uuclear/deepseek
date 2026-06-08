# 变量与控制流

> **前置知识**：无  
> **预计时间**：50 分钟  
> **本章产出**：会用 list/dict、if/for

Python 用缩进表示代码块（4 空格）。

**变量**：`x = 42`、`name = "AI"`
**列表**：`scores = [88, 92, 75]`，索引从 0 开始
**字典**：`user = {"name": "小明", "age": 20}`

**if**：`if score >= 60: ... elif ... else: ...`
**for**：`for i, s in enumerate(scores):`

列表推导：`[s for s in scores if s >= 60]`

## 动手练习

修改 main.py：增加一名学生并打印不及格名单

## 示例文件

- [`examples/part-01-python/01-variables/main.py`](/examples/part-01-python/01-variables/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-01-python/01-variables/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-01-python/02-functions-modules)
