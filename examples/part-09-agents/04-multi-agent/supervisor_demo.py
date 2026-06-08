"""Supervisor + Worker 多 Agent 协作（规则演示）。"""
from __future__ import annotations


def researcher(task: str) -> str:
    return f"[Researcher] 已检索资料：{task} 的相关要点（模拟）。"


def writer(task: str, notes: str) -> str:
    return f"[Writer] 根据「{notes[:40]}...」撰写摘要：{task} 完成。"


def supervisor(goal: str) -> str:
    plan = ["检索背景", "撰写摘要"]
    log = [f"Supervisor 目标: {goal}", f"计划: {plan}"]
    notes = researcher(goal)
    log.append(notes)
    draft = writer(goal, notes)
    log.append(draft)
    log.append("Supervisor: 验收通过，输出最终答案。")
    return "\n".join(log)


def main():
    print(supervisor("向新手介绍 RAG 的三步流程"))


if __name__ == "__main__":
    main()
