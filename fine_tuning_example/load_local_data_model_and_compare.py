from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import inference

model_id = "EleutherAI/pythia-70m"
finetuned_model_dir = "./gitignore_trained_models/pythia-70m-finetuned-cooking_recipes/final"

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(finetuned_model_dir, local_files_only=True)
finetuned_slightly_model.to('cuda')
finetuned_slightly_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir, local_files_only=True)

hf_model = AutoModelForCausalLM.from_pretrained(model_id)
hf_model.to('cuda')
hf_tokenizer = AutoTokenizer.from_pretrained(model_id)

test_dataset = [
	"In a slow cooker, combine all ingredients. Cover and cook on low for 4 hours",
	"In a slow cooker, combine all ingredients. Cover and cook on low for 4 hours or until heated through and cheese is melted. Stir well before serving. Yields 6 servings.",
]

for i in range(1):
	test_question = test_dataset[i]
	print("\n$$$ Question input (test):", test_question)

	print("$$$ Hugging Face model's answer: ")
	print(inference(test_question, hf_model, hf_tokenizer, 1000, 200))
 
	print("$$$ Finetuned slightly model's answer: ")
	print(inference(test_question, finetuned_slightly_model, finetuned_slightly_tokenizer, 1000, 200))
