import logging
import torch
import time
from pprint import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from utilities import tokenize_and_split_data
from inference import inference

logger = logging.getLogger(__name__)

dataset_path = "lamini/lamini_docs"

#model_name = "EleutherAI/pythia-70m"
model_name = "EleutherAI/pythia-410m"

training_config = {
	"model": {
		"pretrained_name": model_name,
		"max_length" : 2048
	},
	"datasets": {
		"use_hf": True,
		"path": dataset_path
	},
	"verbose": True
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

# trim train_dataset and test_dataset to 10 samples
train_dataset = train_dataset
test_dataset = test_dataset

print(train_dataset)
print(test_dataset)

base_model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda")
base_model.to(device)

test_text = test_dataset[0]['question']
print("$$$ Question input (test):", test_text)
print(f"$$$ Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("$$$ Model's answer: ")
print(inference(test_text, base_model, tokenizer))

max_steps = 280 # Set to -1 to train for all steps
num_train_epochs = 1

trained_model_name = f"lamini_docs_{max_steps}_steps_{time.strftime('%Y%m%d_%H%M')}"
output_dir = f"trained_models/{trained_model_name}"

training_args = TrainingArguments(

	# Learning rate
	learning_rate=1.0e-5,

	# Number of training epochs
	num_train_epochs=num_train_epochs,

	# Max steps to train for (each step is a batch of data)
	# Overrides num_train_epochs, if not -1
	max_steps=max_steps,

	# Batch size for training
	per_device_train_batch_size=1,

	# Directory to save model checkpoints
	output_dir=output_dir,

	# Other arguments
	overwrite_output_dir=False, # Overwrite the content of the output directory
	disable_tqdm=False, # Disable progress bars
	eval_steps=120, # Number of update steps between two evaluations
	save_steps=120, # After # steps model is saved
	warmup_steps=1, # Number of warmup steps for learning rate scheduler
	per_device_eval_batch_size=1, # Batch size for evaluation
	evaluation_strategy="steps",
	logging_strategy="steps",
	logging_steps=1,
	optim="adafactor",
	gradient_accumulation_steps = 4,
	gradient_checkpointing=False,

	# Parameters for early stopping
	load_best_model_at_end=True,
	save_total_limit=1,
	metric_for_best_model="eval_loss",
	greater_is_better=False
)

model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      ),
    }
  )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

print("$$$ Start training model...")
train_start = time.time()
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

training_output = trainer.train()

train_end = time.time()
print("$$$ Training time:", train_end - train_start, "seconds")

save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
#save_dir = "trained_models/lamini_docs_280_steps_20240525_1339/final"
print("Saved model to:", save_dir)

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
finetuned_slightly_model.to(device)

for i in range(5):
	test_question = test_dataset[i]['question']
	print("\n$$$ Question input (test):", test_question)

	print("$$$ Finetuned slightly model's answer: ")
	print(inference(test_question, finetuned_slightly_model, tokenizer, 1000, 200))

	test_answer = test_dataset[i]['answer']
	print("$$$ Target answer output (test):", test_answer)
