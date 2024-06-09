from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import inference
from datasets import load_dataset

model_name = "pythia-410m"	# "pythia-70m", "pythia-160m", "pythia-410m"
model_id = f"EleutherAI/{model_name}"
trained_model_name = f"{model_name}_ft_PEFT_cooking"
ft_model_dir = f"gitignore_trained_models/{trained_model_name}/final"

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
#finetuned_slightly_model.to('cuda')
finetuned_slightly_model.load_adapter(ft_model_dir)
finetuned_slightly_tokenizer = AutoTokenizer.from_pretrained(ft_model_dir)

hf_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda')
hf_tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='cuda')

test_dataset = load_dataset("../gitignore_datasets/cooking_recipes", split='train[:5]')

# for i in range(5):
# 	test_question = f"How to make {test_dataset[i]["title"]}?"
# 	print("\n$$$ Question input (test):", test_question)

# 	print("$$$ Hugging Face model's answer: ")
# 	print(inference(test_question, hf_model, hf_tokenizer, 1000, 512))
 
# 	print("$$$ Finetuned slightly model's answer: ")
# 	print(inference(test_question, finetuned_slightly_model, finetuned_slightly_tokenizer, 1000, 512))

# create a loop, input text from console, until I press ctrl+c
while True:
	user_input = input("\n$$$ Enter a question: ")
	if not user_input:
		continue
	print("\n$$$ Hugging Face model's answer:")
	print(inference(user_input, hf_model, hf_tokenizer, 1000, 200))
	print("\n$$$ Finetuned slightly model's answer: ")
	print(inference(user_input, finetuned_slightly_model, finetuned_slightly_tokenizer, 1000, 200))