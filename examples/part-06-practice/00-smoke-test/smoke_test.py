"""DeepSeek / 本地 LLM 部署验收脚本。"""
import os
import sys

from openai import OpenAI

BASE = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
KEY = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

client = OpenAI(base_url=BASE, api_key=KEY or "dummy")
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "回复 OK 两个字母"}],
    max_tokens=10,
)
text = r.choices[0].message.content.strip()
print("回复:", text)
sys.exit(0 if "OK" in text.upper() else 1)
