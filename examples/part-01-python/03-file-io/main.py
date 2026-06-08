from pathlib import Path
p = Path(__file__).resolve().parents[3] / "data" / "sample_corpus.txt"
lines = p.read_text(encoding="utf-8").strip().splitlines()
print(f"共 {len(lines)} 行")
for i, line in enumerate(lines, 1):
    print(f"{i}. {line[:30]}...")
