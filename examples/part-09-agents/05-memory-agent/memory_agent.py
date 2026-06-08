"""短期对话缓冲 + 简易长期记忆（JSON 文件）。"""
from __future__ import annotations

import json
from pathlib import Path

MEM_PATH = Path(__file__).resolve().parent / "memory_store.json"
SHORT_BUFFER: list[dict[str, str]] = []


def load_long_term() -> dict[str, str]:
    if MEM_PATH.exists():
        return json.loads(MEM_PATH.read_text(encoding="utf-8"))
    return {}


def save_long_term(data: dict[str, str]) -> None:
    MEM_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remember(key: str, content: str) -> None:
    store = load_long_term()
    store[key] = content
    save_long_term(store)


def chat_turn(user: str) -> str:
    SHORT_BUFFER.append({"role": "user", "content": user})
    store = load_long_term()
    if "用户名" in user:
        remember("用户名", user.split("叫")[-1].strip())
    name = store.get("用户名", "朋友")
    reply = f"你好 {name}！我记得的长期信息: {list(store.keys())}"
    SHORT_BUFFER.append({"role": "assistant", "content": reply})
    return reply


def main():
    print(chat_turn("我叫小明"))
    print(chat_turn("你还记得我吗？"))
    print("短期缓冲轮数:", len(SHORT_BUFFER))


if __name__ == "__main__":
    main()
