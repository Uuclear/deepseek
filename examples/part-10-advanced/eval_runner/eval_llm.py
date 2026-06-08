"""Golden Set 评测：关键词命中率（无需 API，适合 Agent/LLM 回归冒烟）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "data" / "eval_questions.jsonl"
DOCS_DIR = ROOT / "data" / "sample_docs"


def load_golden() -> list[dict]:
    items = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def score_chunk(query: str, text: str) -> int:
    q_terms = re.findall(r"[\w\u4e00-\u9fff]{2,}", query.lower())
    t_lower = text.lower()
    score = 0
    for term in q_terms:
        if term in t_lower:
            score += len(term)
    # 英文缩写/专有名词补充匹配
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", query):
        if token.lower() in t_lower:
            score += len(token)
    return score


def mock_retrieve(query: str, top_k: int = 4) -> str:
    chunks: list[tuple[int, str, str]] = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        file_score = score_chunk(query, text)
        if file_score:
            chunks.append((file_score + 5, path.name, text[:1200]))
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) < 30:
                continue
            s = score_chunk(query, para)
            if s:
                chunks.append((s, path.name, para))
    chunks.sort(key=lambda x: -x[0])
    if not chunks:
        return "未检索到相关内容"
    seen = set()
    parts = []
    for _, name, body in chunks:
        if name in seen:
            continue
        seen.add(name)
        parts.append(f"[{name}]\n{body}")
        if len(parts) >= top_k:
            break
    return "\n\n".join(parts)


def mock_answer(query: str) -> str:
    return mock_retrieve(query)


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return len(answer) > 20
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits >= max(1, len(keywords) - 1)  # 允许 1 个关键词未命中


def main():
    if not GOLDEN.exists():
        raise SystemExit(f"缺少 Golden Set: {GOLDEN}")

    cases = load_golden()
    passed = 0
    for i, case in enumerate(cases, 1):
        q = case["question"]
        keywords = case.get("expected_keywords", [])
        answer = mock_answer(q)
        ok = keyword_hit(answer, keywords) if keywords else len(answer) > 20
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] #{i} {q[:40]}...")

    rate = passed / max(1, len(cases))
    print(f"\n通过率: {passed}/{len(cases)} ({rate:.1%})")
    return 0 if rate >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
