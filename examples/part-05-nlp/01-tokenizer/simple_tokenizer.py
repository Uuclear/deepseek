import re
from collections import Counter

text = "机器学习很棒！ML is fun."
tokens = re.findall(r"[\w]+|[^\w\s]", text, re.UNICODE)
print("字符级分词:", list(text))
print("简单词级:", tokens)
freq = Counter(tokens)
print("词频 top:", freq.most_common(3))
