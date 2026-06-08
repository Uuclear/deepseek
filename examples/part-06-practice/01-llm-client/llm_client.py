"""可复用的 LLM 聊天封装。"""
import os

from openai import OpenAI

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def chat(
    user: str,
    system: str = "",
    history: list | None = None,
    **kwargs,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    resp = get_client().chat.completions.create(
        model=kwargs.pop("model", os.getenv("LLM_MODEL", "deepseek-chat")),
        messages=messages,
        temperature=kwargs.pop("temperature", 0.3),
        max_tokens=kwargs.pop("max_tokens", 2048),
        **kwargs,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print(chat("用一句话解释什么是机器学习"))
