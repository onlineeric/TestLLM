from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "pythia-160m"  # "pythia-70m", "pythia-160m", "pythia-410m"
model_id = f"EleutherAI/{model_name}"
trained_model_name = f"{model_name}_ft_PEFT_cooking"
output_dir = f"gitignore_trained_models/{trained_model_name}"

hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, device_map='auto')
tokenizer.pad_token = tokenizer.eos_token

print("\n$$$ prepare model for kbit training start...\n")

# Prepare the model for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
	r=8,
	lora_alpha=32,
	target_modules=["attention.query_key_value", "attention.dense"], # target_modules inspected by inspect_model_architecture.py
	lora_dropout=0.05,
	bias="none",
)
model = get_peft_model(model, lora_config)

print("\n$$$ prepare model for kbit training done\n")

# Load the dataset from the local directory
dataset = load_dataset("../gitignore_datasets/cooking_recipes", split='train[:10]')
train_dataset = dataset.select(range(8))
eval_dataset = dataset.select(range(8, 10))
print("\n$$$ load dataset done\n")

# Tokenize the dataset
def formatting_prompts_func(examples):
	text_template = "Tell me how to make {}"
	text_pair_template = """
To make {}, you need the following ingredients:
{}

Cooking directions:
{}"""
	titles = examples["title"]
	ingredients = examples["ingredients"]
	directions = examples["directions"]
	texts = []
	text_pairs = []
	for title, ingredient, direction in zip(titles, ingredients, directions):
		# Must add EOS_TOKEN (tokenizer.eos_token), otherwise your generation will go on forever!
		text = text_template.format(title) + tokenizer.eos_token
		texts.append(text)
		# ingredient and direction is a list in string, convert it to a list and joining the elements
		ingredient_text = ", ".join(eval(ingredient))
		direction_text = "* " + "\n* ".join(eval(direction))
		text_pair = text_pair_template.format(title, ingredient_text, direction_text) + tokenizer.eos_token
		text_pairs.append(text_pair)
	return tokenizer(text=texts, text_pair=text_pairs, truncation=True, padding="max_length", max_length=512)

tokenized_train_datasets = train_dataset.map(formatting_prompts_func, batched=True)
tokenized_eval_datasets = eval_dataset.map(formatting_prompts_func, batched=True)
print("\n$$$ tokenize datasets done\n")

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False,  # Set to True if you are doing masked language modeling
)

# Set up the training arguments
training_args = TrainingArguments(
	output_dir=output_dir,
	eval_strategy="epoch",
	#eval_strategy="no",
	save_strategy="epoch",
	learning_rate=1e-4,
	per_device_train_batch_size=4,
	per_device_eval_batch_size=4,
	#gradient_accumulation_steps=2,  # Simulate a larger batch size
	num_train_epochs=3,
	weight_decay=0.01,
	logging_steps=500,
	load_best_model_at_end=True,
	#fp16=True,  # Enable mixed precision training
)

# Create the Trainer object
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_train_datasets,
	eval_dataset=tokenized_eval_datasets,
	tokenizer=tokenizer,
	data_collator=data_collator,
	callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(f"{output_dir}/final")
