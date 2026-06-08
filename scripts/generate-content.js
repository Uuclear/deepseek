const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

function write(rel, content) {
  const p = path.join(root, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content.trim() + '\n', 'utf8');
}

// --- data ---
write('data/iris.csv', `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.7,3.0,6.1,2.3,virginica
`);

write('data/titanic_sample.csv', `PassengerId,Survived,Pclass,Sex,Age,Fare
1,0,3,male,22,7.25
2,1,1,female,38,71.28
3,1,3,female,26,7.92
4,1,1,female,35,53.1
5,0,3,male,,8.05
6,0,3,male,54,14.5
7,0,1,male,2,51.86
8,0,3,male,27,8.66
`);

write('data/sample_corpus.txt', `机器学习是人工智能的一个分支。
深度学习使用多层神经网络。
Transformer 架构改变了自然语言处理。
DeepSeek 是开源大语言模型系列。
`);

write('examples/requirements.txt', `# Part 0-2 基础
numpy>=1.24
pandas>=2.0

# Part 3 机器学习
scikit-learn>=1.3

# Part 4 深度学习（CPU 可跑）
torch>=2.0
torchvision>=0.15

# Part 5 NLP
# transformers>=4.36  # 可选，注意力可视化示例

# Part 6 DeepSeek 实战
openai>=1.0
python-dotenv>=1.0
# chromadb sentence-transformers  # RAG 项目时安装
`);

// Part 0 examples
write('examples/part-00-env/01-hello/main.py', `print("Hello, AI!")
import sys
print("Python", sys.version.split()[0])
`);

write('examples/part-00-env/01-hello/README.md', `# hello\n\n\`\`\`powershell\npython main.py\n\`\`\`\n`);

write('examples/part-00-env/02-venv-check/check_env.py', `import sys
print("executable:", sys.executable)
try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError:
    print("numpy: not installed (run pip install numpy)")
`);

write('examples/part-00-env/02-venv-check/README.md', `# venv 检查\n\n在激活的虚拟环境中运行 \`python check_env.py\`。\n`);

// Part 1 examples
write('examples/part-01-python/01-variables/main.py', `scores = [88, 92, 75, 63, 95]
avg = sum(scores) / len(scores)
passed = [s for s in scores if s >= 60]
print(f"平均分 {avg:.1f}, 及格 {len(passed)} 人")
for i, s in enumerate(scores):
    grade = "A" if s >= 90 else "B" if s >= 80 else "C" if s >= 60 else "D"
    print(f"  学生{i+1}: {s} -> {grade}")
`);

write('examples/part-01-python/02-functions/utils.py', `def normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
`);

write('examples/part-01-python/02-functions/main.py', `from utils import normalize
data = [10, 20, 30, 40]
print(normalize(data))
`);

write('examples/part-01-python/03-file-io/main.py', `from pathlib import Path
p = Path(__file__).resolve().parents[3] / "data" / "sample_corpus.txt"
lines = p.read_text(encoding="utf-8").strip().splitlines()
print(f"共 {len(lines)} 行")
for i, line in enumerate(lines, 1):
    print(f"{i}. {line[:30]}...")
`);

write('examples/part-01-python/04-numpy-basics/main.py', `import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("a+b=\\n", a + b)
print("a@b=\\n", a @ b)
print("mean:", a.mean(), "std:", a.std())
`);

write('examples/part-01-python/05-pandas-intro/main.py', `import pandas as pd
from pathlib import Path
csv = Path(__file__).resolve().parents[3] / "data" / "iris.csv"
df = pd.read_csv(csv)
print(df.head())
print("\\n按 species 计数:\\n", df["species"].value_counts())
print("\\n数值列均值:\\n", df.select_dtypes("number").mean())
`);

// Part 2 examples
write('examples/part-02-math/01-vectors/main.py', `import numpy as np
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
print("点积 u·v =", u @ v)
print("范数 ||u|| =", np.linalg.norm(u))
A = np.array([[1, 2], [3, 4]])
print("A 的转置:\\n", A.T)
print("det(A) =", np.linalg.det(A))
`);

write('examples/part-02-math/02-gradients/main.py', `import numpy as np

def f(x):
    return x[0]**2 + 3 * x[1]**2

def grad_f(x):
    return np.array([2 * x[0], 6 * x[1]])

x = np.array([3.0, 2.0])
lr = 0.1
for step in range(10):
    g = grad_f(x)
    x = x - lr * g
    print(f"step {step+1}: x={x}, f={f(x):.4f}")
`);

write('examples/part-02-math/03-probability/main.py', `import numpy as np
rng = np.random.default_rng(42)
rolls = rng.integers(1, 7, size=10000)
counts = np.bincount(rolls)[1:]
probs = counts / counts.sum()
print("骰子 1-6 频率:", np.round(probs, 3))
print("理论 1/6 =", round(1/6, 3))
`);

write('examples/part-02-math/04-entropy/main.py', `import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

print("公平硬币 H =", entropy([0.5, 0.5]))
print("偏置 0.9/0.1 H =", round(entropy([0.9, 0.1]), 3))
`);

write('examples/part-02-math/05-loss/main.py', `import numpy as np

y_true = np.array([1, 0, 1, 1])
y_prob = np.array([0.9, 0.2, 0.8, 0.4])

bce = -np.mean(y_true * np.log(y_prob + 1e-9) + (1 - y_true) * np.log(1 - y_prob + 1e-9))
mse = np.mean((y_true - y_prob) ** 2)
print("二元交叉熵:", round(bce, 4))
print("MSE:", round(mse, 4))
`);

// Part 3 examples
write('examples/part-03-ml/01-linear-regression/train.py', `import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
X = df[["petal_length"]].values
y = df["sepal_length"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("coef:", model.coef_[0], "intercept:", model.intercept_)
print("RMSE:", mean_squared_error(y_te, pred, squared=False).__round__(3))
print("R2:", r2_score(y_te, pred).__round__(3))
`);

write('examples/part-03-ml/02-logistic-regression/train.py', `import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
le = LabelEncoder()
y = le.fit_transform(df["species"])
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("accuracy:", accuracy_score(y_te, pred))
print(classification_report(y_te, pred, target_names=le.classes_))
`);

write('examples/part-03-ml/03-decision-tree/train.py', `import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "iris.csv")
le = LabelEncoder()
y = le.fit_transform(df["species"])
X = df.drop(columns=["species"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
print("accuracy:", accuracy_score(y_te, tree.predict(X_te)))
`);

write('examples/part-03-ml/04-metrics/evaluate.py', `from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
y_true = [0, 1, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
print("P/R/F1:", round(p,2), round(r,2), round(f1,2))
print("混淆矩阵:\\n", confusion_matrix(y_true, y_pred))
`);

write('examples/part-03-ml/05-pipeline/train.py', `import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "titanic_sample.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = (df["Sex"] == "female").astype(int)
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500)),
])
scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
print("CV accuracy:", scores.round(3), "mean:", scores.mean().round(3))
`);

// Part 4 examples
write('examples/part-04-dl/01-tensors/main.py', `import torch
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.ones(2, 2)
print("x+y:", x + y)
print("shape:", x.shape, "dtype:", x.dtype)
`);

write('examples/part-04-dl/02-autograd/main.py', `import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print("dy/dx at x=2:", x.grad.item())  # 2*2+3=7
`);

write('examples/part-04-dl/03-mlp/main.py', `import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

model = MLP()
x = torch.randn(4, 1, 28, 28)
print("logits shape:", model(x).shape)
`);

write('examples/part-04-dl/04-cnn/main.py', `import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

print("params:", sum(p.numel() for p in SmallCNN().parameters()))
`);

write('examples/part-04-dl/05-training-loop/train_mnist.py', `"""MNIST 里程碑 — CPU 约 2-5 分钟。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.31,))])
train_ds = datasets.MNIST("./data_mnist", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("./data_mnist", train=False, download=True, transform=tf)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"epoch {epoch+1} train acc: {correct/total:.3f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
print(f"test acc: {correct/total:.3f}")
`);

write('examples/part-04-dl/06-overfitting/demo.py', `import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
X = torch.randn(200, 1)
y = 3 * X.squeeze() + 0.5 * torch.randn(200)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

big = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
small = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))

def fit(model, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()
    with torch.no_grad():
        mse = loss_fn(model(X).squeeze(), y).item()
    return mse

print("大模型 train MSE:", round(fit(big), 4))
print("小模型 train MSE:", round(fit(small), 4))
`);

write('examples/part-04-dl/07-save-load/main.py', `import torch
import torch.nn as nn
from pathlib import Path

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = Tiny()
path = Path(__file__).parent / "tiny.pt"
torch.save({"state_dict": model.state_dict(), "arch": "Tiny"}, path)
ckpt = torch.load(path, weights_only=False)
model2 = Tiny()
model2.load_state_dict(ckpt["state_dict"])
print("loaded ok, sample output shape:", model2(torch.randn(1, 10)).shape)
`);

// Part 5 examples
write('examples/part-05-nlp/01-tokenizer/simple_tokenizer.py', `import re
from collections import Counter

text = "机器学习很棒！ML is fun."
tokens = re.findall(r"[\\w]+|[^\\w\\s]", text, re.UNICODE)
print("字符级分词:", list(text))
print("简单词级:", tokens)
freq = Counter(tokens)
print("词频 top:", freq.most_common(3))
`);

write('examples/part-05-nlp/02-embedding/demo.py', `import numpy as np
vocab = {"我": 0, "爱": 1, "AI": 2}
emb = np.random.randn(len(vocab), 4) * 0.1
sentence = ["我", "爱", "AI"]
vectors = [emb[vocab[w]] for w in sentence]
print("句子向量 shape:", np.stack(vectors).shape)
print("平均池化:", np.mean(vectors, axis=0).round(3))
`);

write('examples/part-05-nlp/03-attention/attention_demo.py', `import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 0., 0.], [0., 1., 0.], [0.5, 0.5, 1.]])
scores = Q @ K.T / np.sqrt(2)
weights = np.array([softmax(scores[0]), softmax(scores[1])])
out = weights @ V
print("attention weights row0:", weights[0].round(3))
print("output shape:", out.shape)
`);

write('examples/part-05-nlp/04-transformer-blocks/demo.py', `"""简化 Transformer 块：LayerNorm + FFN。"""
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x

x = torch.randn(2, 8, 64)
print("out shape:", TransformerBlock()(x).shape)
`);

write('examples/part-05-nlp/05-pretraining-concepts/README.md', `# 预训练概念\n\n阅读 Part 5 第 5 章后，继续 [Part 6 部署](/part-06-practice/00-deployment)。\n`);

// Article template helper
function pubLink(filePath) {
  if (filePath.startsWith('examples/')) return '/' + filePath;
  if (filePath.startsWith('data/')) return '/' + filePath;
  return filePath;
}

function article({ title, prereq, time, output, body, exercise, examples, next }) {
  let ex = '';
  if (examples?.length) {
    ex = '\n## 示例文件\n\n' + examples.map(e => `- [\`${e.path}\`](${pubLink(e.path)}) — ${e.desc}`).join('\n');
    ex += '\n\n运行：在仓库根目录执行 `python ' + (examples[0]?.path || 'examples/...') + '`；构建后可通过 `docs/public/examples/` 下载。';
  }
  return `# ${title}

> **前置知识**：${prereq}  
> **预计时间**：${time}  
> **本章产出**：${output}

${body}

## 动手练习

${exercise}
${ex}

${next ? `\n---\n\n**下一章**：${next}` : ''}
`;
}

const articles = {
  'docs/part-00-env/01-python-install.md': article({
    title: 'Python 与开发环境',
    prereq: '会使用 Windows 文件管理器与浏览器',
    time: '约 45 分钟',
    output: '安装 Python 3.11+、VS Code，能运行第一个脚本',
    body: `## 为什么学 Python

AI 领域绝大多数教程、框架（PyTorch、HuggingFace）和 DeepSeek 示例都以 **Python** 为主。你不需要成为 Python 专家，但要能运行脚本、安装依赖。

## 安装 Python（Windows）

1. 访问 [python.org](https://www.python.org/downloads/) 下载 **3.11 或 3.12**
2. 安装时勾选 **Add python.exe to PATH**
3. 验证：

\`\`\`powershell
python --version
pip --version
\`\`\`

## 安装 VS Code

1. 下载 [VS Code](https://code.visualstudio.com/)
2. 安装扩展：**Python**、**Pylance**
3. 用 VS Code 打开本仓库文件夹

## WSL2（可选，训练推荐）

深度学习训练在 Linux 上更省心。Windows 11 可：

\`\`\`powershell
wsl --install -d Ubuntu
\`\`\`

在 Ubuntu 内同样安装 \`python3-pip\` 与 \`venv\`。

## 第一个脚本

\`\`\`python
print("Hello, AI!")
\`\`\`

保存为 \`main.py\`，在终端运行 \`python main.py\`。
`,
    exercise: `1. 确认 \`python --version\` 输出 3.11+
2. 用 VS Code 打开示例并运行 \`examples/part-00-env/01-hello/main.py\`
3. （可选）在 WSL2 内重复上述步骤`,
    examples: [{ path: 'examples/part-00-env/01-hello/main.py', desc: 'Hello World' }],
    next: '[虚拟环境与 Git](/part-00-env/02-venv-git)',
  }),

  'docs/part-00-env/02-venv-git.md': article({
    title: '虚拟环境与 Git',
    prereq: '已完成 Python 安装',
    time: '约 40 分钟',
    output: '会创建 venv、用 pip 装包、用 Git 克隆/更新仓库',
    body: `## 虚拟环境 venv

每个项目独立依赖，避免版本冲突：

\`\`\`powershell
cd D:\\github\\deepseek
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r examples/requirements.txt
\`\`\`

退出：\`deactivate\`

## requirements.txt

本仓库 Python 示例共用 [\`examples/requirements.txt\`](/examples/requirements.txt)，按 Part 分组注释，可按需分段安装。

## Git 基础

\`\`\`powershell
git clone https://github.com/YOUR_USERNAME/deepseek.git
cd deepseek
git pull
\`\`\`

常用：\`git status\`、\`git add\`、\`git commit\`、\`git push\`

## .gitignore 要点

- 不要提交 \`.env\`（API 密钥）
- 不要提交 \`.venv/\`、\`__pycache__/\`
- 大模型权重用 HuggingFace 缓存，不入库
`,
    exercise: `1. 创建并激活 \`.venv\`
2. \`pip install numpy\` 后运行 \`examples/part-00-env/02-venv-check/check_env.py\`
3. 若未克隆本仓库，练习 \`git clone\` 一次`,
    examples: [{ path: 'examples/part-00-env/02-venv-check/check_env.py', desc: '环境检查' }],
    next: '[Part 1 变量与控制流](/part-01-python/01-variables-control)',
  }),
};

// Generate remaining articles programmatically with content blocks
const more = [
  ['docs/part-01-python/01-variables-control.md', '变量与控制流', '无', '50 分钟', '会用 list/dict、if/for', `Python 用缩进表示代码块（4 空格）。

**变量**：\`x = 42\`、\`name = "AI"\`
**列表**：\`scores = [88, 92, 75]\`，索引从 0 开始
**字典**：\`user = {"name": "小明", "age": 20}\`

**if**：\`if score >= 60: ... elif ... else: ...\`
**for**：\`for i, s in enumerate(scores):\`

列表推导：\`[s for s in scores if s >= 60]\``, '修改 main.py：增加一名学生并打印不及格名单', 'examples/part-01-python/01-variables/main.py', '/part-01-python/02-functions-modules'],
  ['docs/part-01-python/02-functions-modules.md', '函数与模块', '变量与控制流', '45 分钟', '会写函数、拆分模块', `**函数**封装重复逻辑：

\`\`\`python
def normalize(values):
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values]
\`\`\`

**模块**：\`utils.py\` 放工具函数，\`main.py\` 里 \`from utils import normalize\`

**类型提示**（可选）：\`def f(x: float) -> float:\``, '把 normalize 改成 min-max 到 [-1, 1]', 'examples/part-01-python/02-functions/main.py', '/part-01-python/03-file-io'],
  ['docs/part-01-python/03-file-io.md', '文件读写', '函数与模块', '40 分钟', '会读文本/CSV', `\`pathlib.Path\` 跨平台处理路径：

\`\`\`python
from pathlib import Path
text = Path("data/sample_corpus.txt").read_text(encoding="utf-8")
\`\`\`

写文件：\`p.write_text("...", encoding="utf-8")\`

读大文件用 \`with open(...) as f:\` 逐行处理。`, '统计 sample_corpus.txt 每行字数', 'examples/part-01-python/03-file-io/main.py', '/part-01-python/04-numpy-basics'],
  ['docs/part-01-python/04-numpy-basics.md', 'NumPy 入门', '文件 IO', '60 分钟', '会用 ndarray 做向量运算', `NumPy 是数值计算基础，深度学习张量概念与之类似。

\`\`\`python
import numpy as np
a = np.array([[1, 2], [3, 4]])
a.shape  # (2, 2)
a @ b    # 矩阵乘
a.mean(), a.std()
\`\`\`

**广播**：不同 shape 数组可自动对齐运算。`, '创建一个 3×3 单位矩阵并求逆', 'examples/part-01-python/04-numpy-basics/main.py', '/part-01-python/05-pandas-intro'],
  ['docs/part-01-python/05-pandas-intro.md', 'pandas 入门', 'NumPy', '60 分钟', '会读 CSV、做简单统计', `pandas \`DataFrame\` 类似 Excel 表：

\`\`\`python
import pandas as pd
df = pd.read_csv("data/iris.csv")
df.head()
df["species"].value_counts()
df.select_dtypes("number").mean()
\`\`\`

**筛选**：\`df[df["sepal_length"] > 5.5]\``, '读取 titanic_sample.csv，计算各舱位生存率', 'examples/part-01-python/05-pandas-intro/main.py', '/part-02-math/01-vectors-matrices'],
];

for (const [path, title, prereq, time, output, body, exercise, exPath, nextLink] of more) {
  articles[path] = article({
    title, prereq, time, output, body, exercise,
    examples: [{ path: exPath, desc: '本章示例' }],
    next: `[下一章](${nextLink})`,
  });
}

// Part 2
const part2 = [
  ['01-vectors-matrices', '向量与矩阵', 'NumPy 基础', '60 分钟', '理解点积、矩阵乘法', `**向量**是有方向的数列；**矩阵**是二维数组。

- 点积 \`u·v\`：对应元素乘再求和，衡量相似度
- 矩阵乘 \`A@B\`：线性变换组合
- 转置 \`A.T\`、行列式 \`det(A)\`

神经网络每层本质是 \`y = Wx + b\`。`, '用 NumPy 验证 (AB)C = A(BC)', 'examples/part-02-math/01-vectors/main.py', '/part-02-math/02-derivatives-gradients'],
  ['02-derivatives-gradients', '导数与梯度', '向量基础', '60 分钟', '理解梯度下降一步', `**导数**：函数在某点的变化率。\(f(x)=x^2\) 在 \(x=2\) 处导数为 4。

**偏导数**：多变量时对某一变量的导数。
**梯度**：所有偏导组成的向量，指向上升最快方向。

**梯度下降**：\`x = x - lr * grad\`，沿负梯度走以减小损失。`, '对 f(x,y)=x²+y² 手写 5 步梯度下降', 'examples/part-02-math/02-gradients/main.py', '/part-02-math/03-probability'],
  ['03-probability', '概率基础', '高中数学', '45 分钟', '理解频率与概率', `**概率**描述随机事件可能性，取值 0～1。

大数定律：试验次数越多，频率越接近概率。
**条件概率**、**贝叶斯**在分类与 NLP 中常见。

用模拟验证：掷骰子 10000 次，各面频率约 1/6。`, '模拟抛硬币 1000 次，画正面频率', 'examples/part-02-math/03-probability/main.py', '/part-02-math/04-entropy'],
  ['04-entropy', '信息论与熵', '概率基础', '45 分钟', '会算二元熵', `**熵 H**：衡量不确定性。公平硬币 H=1 bit。

公式：\(H = -\\sum p \\log_2 p\)

越确定（如 99%/1%）熵越低。交叉熵常作分类损失。`, '计算四分类均匀分布的熵', 'examples/part-02-math/04-entropy/main.py', '/part-02-math/05-loss-functions'],
  ['05-loss-functions', '损失函数', '熵、梯度', '50 分钟', '理解 MSE 与交叉熵', `**MSE**：回归预测与真值差的平方均值。

**交叉熵**：分类中衡量预测分布与真实分布差异。

训练目标：最小化损失；优化器用梯度更新参数。`, '比较同一组 y 下 BCE 与 MSE 数值', 'examples/part-02-math/05-loss/main.py', '/part-03-ml/01-supervised-unsupervised'],
];

for (const [slug, title, prereq, time, output, body, exercise, exPath, nextLink] of part2) {
  articles[`docs/part-02-math/${slug}.md`] = article({
    title, prereq, time, output, body, exercise,
    examples: [{ path: exPath, desc: '本章示例' }],
    next: `[下一章](${nextLink})`,
  });
}

// Part 3
const part3 = [
  ['01-supervised-unsupervised', '监督与无监督学习', 'pandas、损失函数', '45 分钟', '区分两类机器学习问题', `**监督学习**：有标签 y。分类（物种）、回归（房价）。

**无监督学习**：无标签。聚类、降维。

**流程**：数据 → 特征 → 模型 → 训练 → 评估 → 部署`, '列举 3 个监督与 2 个无监督场景', null, '/part-03-ml/02-linear-regression'],
  ['02-linear-regression', '线性回归', '监督学习概念', '60 分钟', '在 Iris 上拟合线性关系', `模型：\`ŷ = w·x + b\`，最小化 MSE。

sklearn：\`LinearRegression().fit(X, y)\`

看 **系数**、**R²**、**RMSE** 判断拟合好坏。`, '用两个特征预测 sepal_length', 'examples/part-03-ml/01-linear-regression/train.py', '/part-03-ml/03-logistic-regression'],
  ['03-logistic-regression', '逻辑回归', '线性回归', '60 分钟', '多分类准确率 > 90%', `虽名「回归」，用于**分类**。输出经 sigmoid/softmax 变概率。

\`LogisticRegression\` + \`classification_report\` 看 precision/recall。`, '只在 setosa vs rest 上做二分类', 'examples/part-03-ml/02-logistic-regression/train.py', '/part-03-ml/04-decision-trees'],
  ['04-decision-trees', '决策树', '逻辑回归', '50 分钟', '理解决策边界', `树形 if-else 规则，可解释性强。

\`max_depth\` 控制复杂度，过深会过拟合。`, '把 max_depth 改成 10 对比准确率', 'examples/part-03-ml/03-decision-tree/train.py', '/part-03-ml/05-evaluation-metrics'],
  ['05-evaluation-metrics', '评估指标', '分类基础', '45 分钟', '会读混淆矩阵', `**准确率**、**精确率 P**、**召回率 R**、**F1**。

混淆矩阵：TP/FP/FN/TN。

不平衡数据别只看 accuracy。`, '手算一个小混淆矩阵的 F1', 'examples/part-03-ml/04-metrics/evaluate.py', '/part-03-ml/06-sklearn-pipeline'],
  ['06-sklearn-pipeline', 'sklearn 流水线', '评估指标', '60 分钟', 'Pipeline + 交叉验证', `\`Pipeline\` 串联预处理与模型，避免数据泄漏。

\`cross_val_score\` 做 K 折交叉验证更稳。`, '在 Titanic 子集上换 RandomForest', 'examples/part-03-ml/05-pipeline/train.py', '/part-04-dl/01-tensors'],
];

for (const [slug, title, prereq, time, output, body, exercise, exPath, nextLink] of part3) {
  const ex = exPath ? [{ path: exPath, desc: '本章示例' }] : [];
  if (slug === '01-supervised-unsupervised') {
    ex.push({ path: 'data/iris.csv', desc: 'Iris 数据集' });
  }
  articles[`docs/part-03-ml/${slug}.md`] = article({
    title, prereq, time, output, body, exercise, examples: ex,
    next: `[下一章](${nextLink})`,
  });
}

// Part 4
const part4 = [
  ['01-tensors', '张量基础', 'NumPy、Part 3', '45 分钟', 'PyTorch 张量创建与运算', `张量是多维数组，可有 \`requires_grad\`。

\`torch.tensor\`、\`shape\`、\`dtype\`、\`device\`（cpu/cuda）。`, '创建 3×3 随机张量并求和', 'examples/part-04-dl/01-tensors/main.py', '/part-04-dl/02-autograd'],
  ['02-autograd', '自动求导', '张量基础', '50 分钟', '理解 backward()', `PyTorch 自动构建计算图。\`y.backward()\` 后 \`x.grad\` 即 ∂y/∂x。

这是神经网络训练的引擎。`, '对 y=x³ 在 x=1 求导', 'examples/part-04-dl/02-autograd/main.py', '/part-04-dl/03-neural-networks'],
  ['03-neural-networks', '全连接网络', '自动求导', '60 分钟', '定义 MLP', `\`nn.Module\` + \`nn.Linear\` + 激活函数 ReLU。

\`forward\` 定义前向；参数自动注册。`, '把隐藏层改成 256', 'examples/part-04-dl/03-mlp/main.py', '/part-04-dl/04-cnn'],
  ['04-cnn', '卷积神经网络', '全连接网络', '60 分钟', '理解卷积与池化', `CNN 用局部感受野提取图像特征。

\`Conv2d\`、\`MaxPool2d\`、通道数变化。`, '打印每层输出 shape', 'examples/part-04-dl/04-cnn/main.py', '/part-04-dl/05-training-loop'],
  ['05-training-loop', '训练循环', 'CNN', '90 分钟', '完整 train/eval 循环', `标准四步：\`zero_grad\` → \`forward\` → \`loss.backward\` → \`step\`

DataLoader 批处理；\`model.train()\` / \`model.eval()\`。`, '把 epoch 改成 5 观察 test acc', 'examples/part-04-dl/05-training-loop/train_mnist.py', '/part-04-dl/06-overfitting'],
  ['06-overfitting', '过拟合与正则', '训练循环', '45 分钟', '认识过拟合', `模型过大或数据过少 → 训练好、测试差。

**对策**：更多数据、Dropout、权重衰减、早停。`, '对比 demo 大小模型 MSE', 'examples/part-04-dl/06-overfitting/demo.py', '/part-04-dl/07-save-load'],
  ['07-save-load', '保存与加载 · MNIST', '过拟合', '90 分钟', 'MNIST test acc > 95%', `**里程碑**：运行 [\`train_mnist.py\`](/examples/part-04-dl/05-training-loop/train_mnist.py) 完整训练。

\`torch.save\` / \`load_state_dict\` 持久化权重。`, '保存 MNIST 模型并加载推理一张图', 'examples/part-04-dl/07-save-load/main.py', '/part-05-nlp/01-tokenization-embedding'],
];

for (const [slug, title, prereq, time, output, body, exercise, exPath, nextLink] of part4) {
  const ex = [{ path: exPath, desc: '本章示例' }];
  if (slug === '07-save-load') {
    ex.unshift({ path: 'examples/part-04-dl/05-training-loop/train_mnist.py', desc: 'MNIST 训练（里程碑）' });
  }
  articles[`docs/part-04-dl/${slug}.md`] = article({
    title, prereq, time, output, body, exercise, examples: ex,
    next: `[下一章](${nextLink})`,
  });
}

// Part 5
const part5 = [
  ['01-tokenization-embedding', '分词与 Embedding', 'PyTorch 基础', '60 分钟', '文本→token→向量', `**分词**把文本切成 token（字/词/子词 BPE）。

**Embedding** 把离散 id 映射为稠密向量，语义相近则向量接近。`, '统计 sample_corpus 唯一词数', 'examples/part-05-nlp/01-tokenizer/simple_tokenizer.py', '/part-05-nlp/02-rnn-intuition'],
  ['02-rnn-intuition', 'RNN 直觉', 'Embedding', '45 分钟', '理解序列建模', `RNN 按时间步传递隐状态，适合序列。

缺点：长序列梯度消失；被 Transformer 广泛取代但仍值得理解。`, '画一张 3 步 RNN 展开图（纸笔）', 'examples/part-05-nlp/02-embedding/demo.py', '/part-05-nlp/03-attention'],
  ['03-attention', '注意力机制', 'RNN 直觉', '60 分钟', '手算 tiny attention', `Attention：Query 对 Key 算权重，加权求和 Value。

\`softmax(QK^T / sqrt(d)) V\` — Transformer 核心。`, '修改 demo 里 Q 向量看权重变化', 'examples/part-05-nlp/03-attention/attention_demo.py', '/part-05-nlp/04-transformer'],
  ['04-transformer', 'Transformer 结构', 'Attention', '75 分钟', '认识 Encoder 块', `多层：**Self-Attention** → Add&Norm → **FFN** → Add&Norm。

\`nn.MultiheadAttention\` 已实现多头注意力。`, '运行 demo 打印输出 shape', 'examples/part-05-nlp/04-transformer-blocks/demo.py', '/part-05-nlp/05-pretraining'],
  ['05-pretraining', '预训练与 LLM 衔接', 'Transformer', '60 分钟', '理解预训练→微调→推理', `**预训练**：海量文本自监督学语言表示。

**微调 SFT**：用指令数据对齐行为。

**推理**：Chat API 就是部署后的预训练+对齐模型。

👉 读完本章，进入 [Part 6 DeepSeek 部署](/part-06-practice/00-deployment)。`, '列出预训练/微调/推理各 1 个例子', 'examples/part-05-nlp/05-pretraining-concepts/README.md', null],
];

for (const [slug, title, prereq, time, output, body, exercise, exPath, nextLink] of part5) {
  articles[`docs/part-05-nlp/${slug}.md`] = article({
    title, prereq, time, output, body, exercise,
    examples: [{ path: exPath, desc: '本章示例' }],
    next: nextLink ? `[下一章](${nextLink})` : undefined,
  });
}

for (const [rel, content] of Object.entries(articles)) {
  write(rel, content);
}

// README stubs for example dirs
const readmes = [
  'examples/part-01-python/01-variables/README.md',
  'examples/part-02-math/01-vectors/README.md',
  'examples/part-03-ml/01-linear-regression/README.md',
  'examples/part-04-dl/05-training-loop/README.md',
  'examples/part-05-nlp/03-attention/README.md',
];
for (const r of readmes) {
  write(r, `# 示例\n\n\`\`\`powershell\npython main.py\n# 或 python train.py\n\`\`\`\n`);
}

console.log('Generated', Object.keys(articles).length, 'articles + examples + data');
