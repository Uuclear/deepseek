scores = [88, 92, 75, 63, 95]
avg = sum(scores) / len(scores)
passed = [s for s in scores if s >= 60]
print(f"平均分 {avg:.1f}, 及格 {len(passed)} 人")
for i, s in enumerate(scores):
    grade = "A" if s >= 90 else "B" if s >= 80 else "C" if s >= 60 else "D"
    print(f"  学生{i+1}: {s} -> {grade}")
