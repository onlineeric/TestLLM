from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
import os
from peft import LoraConfig


model_name = "Phi-3-mini-4k-instruct"
model_id = f"microsoft/{model_name}"
trained_model_name = f"{model_name}_PEFT_cooking"
output_dir = f"gitignore_trained_models/{trained_model_name}"

hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, device_map='auto')
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
	lora_alpha=16,
	lora_dropout=0.1,
	r=64,
	bias="none",
	task_type="CAUSAL_LM",
	target_modules=["transformer.lm_head"],
)
model.add_adapter(peft_config, adapter_name="peft_adapter")

# Load the dataset from the local directory
total_records = 50
train_records = int(total_records * 0.9)
dataset = load_dataset("../gitignore_datasets/cooking_recipes", split=f'train[:{total_records}]')
train_dataset = dataset.select(range(train_records))
eval_dataset = dataset.select(range(train_records, total_records))
print("\n$$$ load dataset done\n")

# Tokenize the dataset
def formatting_prompts_func(examples):
	text_template = "Tell me how to make {}"
	text_pair_template = """
To make {}, you need the following ingredients:
{}

Cooking directions to make {}:
{}"""
	titles			= examples["title"]
	ingredients		= examples["ingredients"]
	directions		= examples["directions"]
	texts = []
	text_pairs = []
	for title, ingredient, direction in zip(titles, ingredients, directions):
		# Must add EOS_TOKEN (tokenizer.eos_token), otherwise your generation will go on forever!
		text = text_template.format(title) + tokenizer.eos_token
		texts.append(text)
		# ingredient and direction is a list in string, convert it to a list and joining the elements
		ingredient_text = "* " + "\n* ".join(eval(ingredient))
		direction_text = "* " + "\n* ".join(eval(direction))
		text_pair = text_pair_template.format(title, ingredient_text, title, direction_text) + tokenizer.eos_token
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
batch_size = 4
epochs = 3
training_args = TrainingArguments(
	output_dir=output_dir,
	# eval_strategy="epoch",
	# save_strategy="epoch",
	eval_strategy="steps",  # Evaluate at each logging step
	eval_steps=int(train_records/batch_size*epochs/10),  # Evaluate every every 10% of the training data
	save_strategy="steps",
	save_steps=int(train_records/batch_size*epochs/10),  # Save checkpoint every 10% of the training data
	logging_steps=int(train_records/batch_size*epochs/20), # Log every 5% of the training data
	learning_rate=1e-4,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	#gradient_accumulation_steps=2,  # Simulate a larger batch size
	num_train_epochs=epochs,
	weight_decay=0.01,
	#load_best_model_at_end=True,
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
	#callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping
)

# Fine-tune the model
print("\n$$$ start training\n")
trainer.train()

# Save the fine-tuned model
final_model_dir = f"{output_dir}/final"
print(f"\n$$$ Saving model to {final_model_dir} \n")
#trainer.save_model(final_model_dir)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
