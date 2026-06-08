"""简化知识蒸馏 demo：随机 Teacher 指导小 MLP Student（教学用，无需大模型）。"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Teacher(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def distill_loss(student_logits, teacher_logits, labels, T: float = 4.0, alpha: float = 0.7):
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    dim = 32
    n = 2000
    x = torch.randn(n, dim)
    teacher = Teacher(dim)
    with torch.no_grad():
        labels = teacher(x).argmax(dim=-1)

    loader = DataLoader(TensorDataset(x, labels), batch_size=args.batch_size, shuffle=True)
    student = Student(dim)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    teacher.eval()
    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0.0
        correct = 0
        count = 0
        for bx, by in loader:
            with torch.no_grad():
                t_logits = teacher(bx)
            s_logits = student(bx)
            loss = distill_loss(s_logits, t_logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            correct += (s_logits.argmax(-1) == by).sum().item()
            count += by.size(0)
        acc = correct / count
        print(f"epoch {epoch}/{args.epochs} loss={total/len(loader):.4f} acc={acc:.3f}")

    print("蒸馏 demo 完成。Student 学习了 Teacher 的软分布 + 硬标签。")


if __name__ == "__main__":
    main()
