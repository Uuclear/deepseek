# 虚拟环境与 Git

> **前置知识**：已完成 Python 安装  
> **预计时间**：约 40 分钟  
> **本章产出**：会创建 venv、用 pip 装包、用 Git 克隆/更新仓库

## 虚拟环境 venv

每个项目独立依赖，避免版本冲突：

```powershell
cd D:\github\deepseek
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r examples/requirements.txt
```

退出：`deactivate`

## requirements.txt

本仓库 Python 示例共用 [`examples/requirements.txt`](/examples/requirements.txt)，按 Part 分组注释，可按需分段安装。

## Git 基础

```powershell
git clone https://github.com/YOUR_USERNAME/deepseek.git
cd deepseek
git pull
```

常用：`git status`、`git add`、`git commit`、`git push`

## .gitignore 要点

- 不要提交 `.env`（API 密钥）
- 不要提交 `.venv/`、`__pycache__/`
- 大模型权重用 HuggingFace 缓存，不入库


## 动手练习

1. 创建并激活 `.venv`
2. `pip install numpy` 后运行 `examples/part-00-env/02-venv-check/check_env.py`
3. 若未克隆本仓库，练习 `git clone` 一次

## 示例文件

- [`examples/part-00-env/02-venv-check/check_env.py`](/examples/part-00-env/02-venv-check/check_env.py) — 环境检查

运行：在仓库根目录执行 `python examples/part-00-env/02-venv-check/check_env.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[Part 1 变量与控制流](/part-01-python/01-variables-control)
