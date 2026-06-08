"""轻量 RAG Agent：关键词检索 + 可选 LLM（无 API 时仅返回检索片段）。"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "data" / "sample_docs"


def load_chunks() -> list[tuple[str, str]]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*")):
        text = path.read_text(encoding="utf-8")
        name = path.name
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            para = para.strip()
            if len(para) > 30:
                chunks.append((f"{name}#chunk{i}", para))
    return chunks


def search(query: str, top_k: int = 3) -> list[tuple[str, str, int]]:
    q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    scored = []
    for doc_id, text in load_chunks():
        t_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
        score = len(q_tokens & t_tokens)
        if score:
            scored.append((doc_id, text, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


def answer_with_llm(query: str, context: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"[仅检索模式]\n{context}"
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    prompt = f"根据以下资料回答问题，并注明来源。\n\n资料:\n{context}\n\n问题: {query}"
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def main():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    query = "什么是 ReAct Agent？"
    hits = search(query)
    ctx = "\n\n".join(f"[{d}]\n{t}" for d, t, _ in hits)
    print("检索结果:\n", ctx)
    print("\n--- 回答 ---\n", answer_with_llm(query, ctx))


if __name__ == "__main__":
    main()
