import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

#model_id = "microsoft/Phi-3-mini-4k-instruct"
model_id = "microsoft/Phi-3-mini-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
	model_id,
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
	{"role": "system", "content": "Your are a Chef providing Cooking Recipes."},
	{"role": "user", "content": {}},
]

pipe = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
)

generation_args = {
	"max_new_tokens": 2400,
	"return_full_text": False,
	"temperature": 0.3,
	"do_sample": False,
}

while True:
	input_str = input("Enter your message: ")
	messages[1]["content"] = input_str
	start_time = time.time()
	output = pipe(messages, **generation_args)
	end_time = time.time()
	print("\nTime taken: ", end_time - start_time, "seconds")
	print("\nResponse: ", output[0]['generated_text'])
