from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load the pre-trained model and tokenizer
model_id = "EleutherAI/pythia-70m"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)



# Load the dataset from the local directory
dataset = load_dataset("./gitignore_datasets/cooking_recipes", split="train")

# Tokenize the dataset
def tokenize_function(examples):
	return tokenizer(examples["directions"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

trained_model_name = "pythia-70m-finetuned-cooking_recipes"
output_dir = f"gitignore_trained_models/{trained_model_name}"

# Set up the training arguments
training_args = TrainingArguments(
	output_dir=output_dir,
	evaluation_strategy="epoch",
	learning_rate=2e-5,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	num_train_epochs=3,
	weight_decay=0.01,
	logging_steps=500,
)

# Create the Trainer object
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_datasets["train"],
	eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(f"{output_dir}/final")
