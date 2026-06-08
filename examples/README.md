# 示例代码

与 `docs/` 各 Part 章节一一对应，可在仓库根目录直接运行。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r examples/requirements.txt
```

## 目录

| 目录 | 说明 |
|------|------|
| `part-00-env/` | 环境检查 |
| `part-01-python/` | Python 基础 |
| `part-02-math/` | 数学直觉 |
| `part-03-ml/` | sklearn 机器学习 |
| `part-04-dl/` | PyTorch 深度学习（含 MNIST） |
| `part-05-nlp/` | NLP / Attention |
| `part-06-practice/` | DeepSeek smoke test、llm_client |

构建网站时，`scripts/copy-examples.js` 会将本目录复制到 `docs/public/examples/` 供下载。
