import transformers, os, torch
from pipeline_runner_instruct import run_pipeline

model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
hf_token = os.getenv('HUGGINGFACE_TOKEN')

messages = [
		{"role": "system", "content": "You are a chatbot."},
		{"role": "user", "content": "Tell me how to train my dog to sit?"},
		{"role": "system", "content": "To train your dog to sit, follow these steps:"},
		{"role": "system", "content": "1. Hold a treat close to your dog's nose."},
		{"role": "system", "content": "2. Move your hand up, allowing your dog's head to follow the treat and causing their bottom to lower."},
		{"role": "system", "content": "3. Once they are in sitting position, say 'sit' and give them the treat."},
		{"role": "system", "content": "4. Repeat this process several times a day until your dog has mastered the command."},
		{"role": "user", "content": "My dog is not responding to the 'sit' command. What should I do?"},
]

# pipeline = transformers.pipeline(
# 		"text-generation",
# 		token=hf_token,
# 		model=model_id,
# 		model_kwargs={"torch_dtype": torch.bfloat16},
# 		device=-1, # CPU
# 		)

# run_pipeline(pipeline, messages, 500) 

pipeline = transformers.pipeline(
		"text-generation",
		token=hf_token,
		model=model_id,
		model_kwargs={"torch_dtype": torch.bfloat16},
		#device_map="auto",
		)

run_pipeline(pipeline, messages, 1000)
