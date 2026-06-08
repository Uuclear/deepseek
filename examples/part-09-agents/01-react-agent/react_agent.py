"""ReAct 循环演示（mock 工具，无需 API）。"""
from __future__ import annotations

import re
from typing import Callable

MOCK_WEATHER = {"北京": "晴 25°C", "上海": "多云 22°C", "深圳": "阵雨 28°C"}


def get_weather(city: str) -> str:
    return MOCK_WEATHER.get(city, f"{city}：暂无数据（模拟）")


TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
}


def run_react(question: str, max_steps: int = 5) -> str:
    """简化 ReAct：用规则模拟 Thought/Action/Observation。"""
    context = f"Question: {question}\n"
    for step in range(1, max_steps + 1):
        if "天气" in question and step == 1:
            thought = "需要查询城市天气"
            action = 'get_weather(city="北京")'
            obs = get_weather("北京")
        elif step == 1:
            thought = "无需工具，直接回答"
            return context + f"Thought: {thought}\nFinal Answer: 这是教学演示 Agent。\n"
        else:
            break
        context += f"Thought {step}: {thought}\n"
        context += f"Action {step}: {action}\n"
        context += f"Observation {step}: {obs}\n"
    context += "Final Answer: 北京今天天气为 " + get_weather("北京") + "\n"
    return context


def main():
    q = "北京今天天气怎么样？"
    print(run_react(q))
    print("---")
    print(run_react("什么是机器学习？"))


if __name__ == "__main__":
    main()
