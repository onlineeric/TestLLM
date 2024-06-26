from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
import os

model_name = "pythia-160m"	# "pythia-70m", "pythia-160m", "pythia-410m"
model_id = f"EleutherAI/{model_name}"
trained_model_name = f"{model_name}_ft_cooking"
output_dir = f"gitignore_trained_models/{trained_model_name}"

hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset from the local directory
dataset = load_dataset("../gitignore_datasets/cooking_recipes", split='train[:1000]')
train_dataset = dataset.select(range(800))
eval_dataset = dataset.select(range(800, 1000))
print("\n$$$ load dataset done\n")

# Tokenize the dataset
def formatting_prompts_func(examples):
	alpaca_prompt = """
	### Instruction:
	{}

	### Input:
	{}

	### Response:
	{}"""

	response_template = """	Ingredients: {}
	Steps: {}"""

	titles			= examples["title"]
	ingredients		= examples["ingredients"]
	directions		= examples["directions"]
	texts = []
	for title, ingredient, direction in zip(titles, ingredients, directions):
		# Must add EOS_TOKEN (tokenizer.eos_token), otherwise your generation will go on forever!
		text = alpaca_prompt.format(
			"Tell me how to make this dish. Please include the ingredients and the steps to make it.",
			title,
			response_template.format(ingredient, direction)
			) + tokenizer.eos_token
		texts.append(text)
	return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

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
	learning_rate=2e-5,
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
