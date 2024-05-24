import itertools
import jsonlines

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
finetuning_dataset = []

print("Pretrained dataset:")
for i in top_n:
  text_with_prompt_template = prompt_template_qa.format(question=i["question"], answer=i["answer"])
  print(text_with_prompt_template)
  finetuning_dataset.append(text_with_prompt_template)

# Save the processed dataset
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
  writer.write_all(finetuning_dataset)