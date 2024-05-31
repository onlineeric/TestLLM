from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Load the dataset from local storage
dataset_path = "./gitignore_datasets/cooking_recipes"
dataset = load_from_disk(dataset_path)

# Ensure the dataset has the necessary columns
required_columns = ['title', 'ingredients', 'directions', 'NER']
if not all(column in dataset['train'].column_names for column in required_columns):
	raise ValueError("Dataset must contain the columns: 'title', 'ingredients', 'directions', 'NER'")

# Step 2: Load the model and tokenizer
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(examples):
	combined_text = examples['title'] + " " + examples['ingredients'] + " " + examples['directions'] + " " + examples['NER']
	return tokenizer(combined_text, truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Set up the data collator
data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False,  # Set to True if you are doing masked language modeling
)

trained_model_name = f"CookingRecipes_{time.strftime('%Y%m%d_%H%M')}"
output_dir = f"gitignore_trained_models/{trained_model_name}"

# Step 5: Set up the Trainer and TrainingArguments
training_args = TrainingArguments(
	output_dir=output_dir,
	evaluation_strategy='epoch',
	learning_rate=2e-5,
	per_device_train_batch_size=4,
	per_device_eval_batch_size=4,
	num_train_epochs=3,
	weight_decay=0.01,
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_datasets['train'],
	data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
save_dir = f'{output_dir}/final'
trainer.save_model(save_dir)