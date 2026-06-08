# Python 与开发环境

> **前置知识**：会使用 Windows 文件管理器与浏览器  
> **预计时间**：约 45 分钟  
> **本章产出**：安装 Python 3.11+、VS Code，能运行第一个脚本

## 为什么学 Python

AI 领域绝大多数教程、框架（PyTorch、HuggingFace）和 DeepSeek 示例都以 **Python** 为主。你不需要成为 Python 专家，但要能运行脚本、安装依赖。

## 安装 Python（Windows）

1. 访问 [python.org](https://www.python.org/downloads/) 下载 **3.11 或 3.12**
2. 安装时勾选 **Add python.exe to PATH**
3. 验证：

```powershell
python --version
pip --version
```

## 安装 VS Code

1. 下载 [VS Code](https://code.visualstudio.com/)
2. 安装扩展：**Python**、**Pylance**
3. 用 VS Code 打开本仓库文件夹

## WSL2（可选，训练推荐）

深度学习训练在 Linux 上更省心。Windows 11 可：

```powershell
wsl --install -d Ubuntu
```

在 Ubuntu 内同样安装 `python3-pip` 与 `venv`。

## 第一个脚本

```python
print("Hello, AI!")
```

保存为 `main.py`，在终端运行 `python main.py`。


## 动手练习

1. 确认 `python --version` 输出 3.11+
2. 用 VS Code 打开示例并运行 `examples/part-00-env/01-hello/main.py`
3. （可选）在 WSL2 内重复上述步骤

## 示例文件

- [`examples/part-00-env/01-hello/main.py`](/examples/part-00-env/01-hello/main.py) — Hello World

运行：在仓库根目录执行 `python examples/part-00-env/01-hello/main.py`；构建后可通过 `docs/public/examples/` 下载。


---

**下一章**：[虚拟环境与 Git](/part-00-env/02-venv-git)
