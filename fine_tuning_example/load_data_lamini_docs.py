import itertools

from datasets import load_dataset

pretrained_dataset = load_dataset("lamini/lamini_docs", split="train", streaming=True)

n = 10
top_n = itertools.islice(pretrained_dataset, n)
# for i in top_n:
#   print("$$$ Question: ", i["question"])
#   print("$$$ Answer: ", i["answer"])

prompt_template_qa = """
### Question:
{question}

### Answer:
{answer}
"""

print("Pretrained dataset:")
for i in top_n:
  text_with_prompt_template = prompt_template_qa.format(question=i["question"], answer=i["answer"])
  print(text_with_prompt_template)
  
