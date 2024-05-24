from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from inference import inference

model_id = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
print("$$$ finetuning_dataset:")
print(finetuning_dataset)

test_sample = finetuning_dataset["test"][0]
print("$$$ test_sample:")
print(test_sample)

print("$$$ inferenced text:")
print(inference(test_sample["question"], model, tokenizer))

# compare to finetuned model, lamini/lamini_docs_finetuned was finetuned on EleutherAI/pythia-70m
model_id = "lamini/lamini_docs_finetuned"
instruction_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(inference(test_sample["question"], instruction_model, tokenizer))
