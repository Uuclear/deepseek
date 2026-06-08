# 文件读写

> **前置知识**：函数与模块  
> **预计时间**：40 分钟  
> **本章产出**：会读文本/CSV

`pathlib.Path` 跨平台处理路径：

```python
from pathlib import Path
text = Path("data/sample_corpus.txt").read_text(encoding="utf-8")
```

写文件：`p.write_text("...", encoding="utf-8")`

读大文件用 `with open(...) as f:` 逐行处理。

## 动手练习

统计 sample_corpus.txt 每行字数

## 示例文件

- [`examples/part-01-python/03-file-io/main.py`](/examples/part-01-python/03-file-io/main.py) — 本章示例

运行：在仓库根目录执行 `python examples/part-01-python/03-file-io/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[下一章](/part-01-python/04-numpy-basics)
