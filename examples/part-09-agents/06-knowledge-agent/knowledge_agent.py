"""个人知识库 Agent：检索 data/sample_docs + 可选 DeepSeek API。"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "data" / "sample_docs"


def load_chunks() -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            para = para.strip()
            if len(para) > 40:
                chunks.append((f"{path.name}#chunk{i}", para))
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
    return scored[:top_k] if scored else [(c[0], c[1], 0) for c in load_chunks()[:top_k]]


def answer_with_llm(query: str, context: str) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "[mock 模式 — 未设置 API Key]\n\n"
            f"问题: {query}\n\n"
            f"检索到的资料:\n{context}\n\n"
            "提示: 设置 DEEPSEEK_API_KEY 后可获得 LLM 整合回答。"
        )
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    prompt = (
        "你是个人知识库助手。仅根据以下资料回答问题，并注明来源文件名。\n\n"
        f"资料:\n{context}\n\n问题: {query}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def run(query: str, top_k: int = 3) -> str:
    hits = search(query, top_k=top_k)
    ctx = "\n\n".join(f"[来源: {doc_id}]\n{text}" for doc_id, text, _ in hits)
    print("=== 检索结果 ===")
    print(ctx)
    print("\n=== 回答 ===")
    answer = answer_with_llm(query, ctx)
    print(answer)
    return answer


def main():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="个人知识库 Agent")
    parser.add_argument("--query", type=str, default="什么是 BPE 分词？")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
