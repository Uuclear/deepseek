# 从零开始学 AI · DeepSeek 文档集

VitePress 静态学习网站：Python → 数学 → ML → DL → NLP → DeepSeek 实战。

## 本地开发

```powershell
npm install
npm run dev
```

浏览器打开 `http://localhost:5173/deepseek/`

## 构建与部署

```powershell
npm run build
npm run preview
```

推送到 `main` 分支后，GitHub Actions 自动部署到 GitHub Pages（`base: /deepseek/`）。

## 目录结构

```text
docs/           # VitePress 文档（Part 0～7）
examples/       # 可运行 Python 示例
data/           # 小型示例数据集
scripts/        # 构建辅助脚本
```

## 快速入口

| 路径 | 说明 |
|------|------|
| [学习首页](docs/index.md) | Hero + 课程结构 |
| [学习路线](docs/roadmap.md) | L0～L6 能力阶梯 |
| [Part 6 实战](docs/part-06-practice/00-deployment.md) | DeepSeek 部署 |
| [Part 7 原理](docs/part-07-theory/training-guide.md) | 训练全流程 |

根目录 `guide-*.md` 与 `deepseek-*.md` 已迁移为 stub，指向 `docs/` 内新路径。

## Python 示例

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r examples/requirements.txt
python examples/part-06-practice/00-smoke-test/smoke_test.py
```
