"""DeepSeek Function Calling 演示（需 API Key）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
TOOLS_PATH = ROOT / "data" / "agent_tools.json"


def mock_get_weather(city: str) -> str:
    data = {"北京": "晴 25°C", "上海": "多云 22°C"}
    return data.get(city, f"{city}：模拟天气数据")


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("请复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY")
        return

    tools = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
    client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    messages = [{"role": "user", "content": "北京现在天气如何？请用工具查询。"}]
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=messages,
        tools=tools[:1],
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments)
        result = mock_get_weather(args.get("city", "北京"))
        messages.append(msg)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
        )
        final = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=messages,
            tools=tools[:1],
        )
        print(final.choices[0].message.content)
    else:
        print(msg.content)


if __name__ == "__main__":
    main()
