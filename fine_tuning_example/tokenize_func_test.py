from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from pprint import pprint

# Load the pre-trained model and tokenizer
model_id = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset from the local directory
dataset = load_dataset("../gitignore_datasets/cooking_recipes", split='train[:5]')
print("$$$ dataset:")
pprint(dataset, indent=4)

# Tokenize the dataset
def tokenize_function(examples):
	print(f"\n$$$ examples:")
	pprint(examples, indent=4)
	tokenized_data = tokenizer(examples["directions"], truncation=True, max_length=2048)
	return tokenized_data

tokenized_datasets = dataset.map(tokenize_function, batched=True)
