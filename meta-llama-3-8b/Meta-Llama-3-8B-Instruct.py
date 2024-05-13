import transformers, os, torch
from pipeline_runner_instruct import run_pipeline

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = os.getenv('HUGGINGFACE_TOKEN')

messages = [
		{"role": "system", "content": "You are a chatbot."},
		{"role": "user", "content": "How to train my dog to sit?"},
]

pipeline = transformers.pipeline(
		"text-generation",
		token=hf_token,
		model=model_id,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device=-1,
		)

run_pipeline(pipeline, messages) 

pipeline = transformers.pipeline(
		"text-generation",
		token=hf_token,
		model=model_id,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map="auto",
		)

run_pipeline(pipeline, messages) 
