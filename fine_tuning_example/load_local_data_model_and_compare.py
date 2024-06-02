from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import inference
from datasets import load_dataset

model_id = "EleutherAI/pythia-70m"
finetuned_model_dir = "./gitignore_trained_models/pythia-70m-finetuned-cooking_recipes/final"

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(finetuned_model_dir, local_files_only=True)
finetuned_slightly_model.to('cuda')
finetuned_slightly_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir, local_files_only=True)

hf_model = AutoModelForCausalLM.from_pretrained(model_id)
hf_model.to('cuda')
hf_tokenizer = AutoTokenizer.from_pretrained(model_id)

test_dataset = load_dataset("../gitignore_datasets/cooking_recipes", split='train[:5]')

for i in range(3):
	test_question = f"How to make {test_dataset[i]["title"]}?"
	print("\n$$$ Question input (test):", test_question)

	print("$$$ Hugging Face model's answer: ")
	print(inference(test_question, hf_model, hf_tokenizer, 1000, 200))
 
	print("$$$ Finetuned slightly model's answer: ")
	print(inference(test_question, finetuned_slightly_model, finetuned_slightly_tokenizer, 1000, 500))
