from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback

# Load the pre-trained model and tokenizer
trained_model_name = "pythia-70m-finetuned-cooking_recipes"
#trained_model_name = "pythia-410m-finetuned-cooking_recipes"
output_dir = f"gitignore_trained_models/{trained_model_name}"
model_id = "EleutherAI/pythia-70m"
#model_id = "EleutherAI/pythia-410m"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset from the local directory
dataset = load_dataset("./gitignore_datasets/cooking_recipes", split='train[:500]')
train_dataset = dataset.select(range(400))
eval_dataset = dataset.select(range(400, 500))
print("$$$ load dataset done")

# Tokenize the dataset
def tokenize_function(examples):
	return tokenizer(examples["directions"], truncation=True, max_length=2048)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["title", "ingredients", "NER", "link", "source"])
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True, remove_columns=["title", "ingredients", "NER", "link", "source"])

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
	num_train_epochs=3,
	weight_decay=0.01,
	logging_steps=500,
	load_best_model_at_end=True,	
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
