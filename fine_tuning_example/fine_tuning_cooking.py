from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import time

# Load the dataset
dataset = load_dataset('CodeKapital/CookingRecipes')

# Select only the first 1000 samples
dataset = dataset['train'].select(range(10))

# Load the tokenizer and model
model_name = "EleutherAI/pythia-70m"
#model_name = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
	# return tokenizer(examples['title'] + " " + examples['ingredients'] + " " + examples['directions'], padding="max_length", truncation=True, max_length=512)
	# Ensure all necessary keys are present and not empty
	title = examples.get('title', "")
	print('$$$ title: ', title)
	ingredients = examples.get('ingredients', "")
	directions = examples.get('directions', "")
	
	# Combine the text fields
	text = title + " " + ingredients + " " + directions
	
	# Tokenize the combined text
	return tokenizer(text, padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare for training
train_dataset = tokenized_datasets["train"]

# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

trained_model_name = f"CookingRecipes_{time.strftime('%Y%m%d_%H%M')}"
output_dir = f"trained_models/{trained_model_name}"

# Define the training arguments
training_args = TrainingArguments(
	output_dir=output_dir,
	evaluation_strategy="epoch",
	learning_rate=2e-5,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	num_train_epochs=3,
	weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
# model.save_pretrained("./trained_model")
# tokenizer.save_pretrained("./trained_model")
save_dir = f'{output_dir}/final'
trainer.save_model(save_dir)
