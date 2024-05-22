import itertools

from datasets import load_dataset

pretrained_dataset = load_dataset("lamini/lamini_docs", split="train", streaming=True)

n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)
