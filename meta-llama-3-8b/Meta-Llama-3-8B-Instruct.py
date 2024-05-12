import transformers, time, json, os, torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_pipline_time = time.time()

pipeline = transformers.pipeline(
		"text-generation",
		token=hf_token,
		model=model_id,
		model_kwargs={"torch_dtype": torch.bfloat16},
		device_map="auto",
)

end_pipeline_time = time.time()
print("$$$$$$$$$$$$$$$$$ Time taken for pipeline: ", end_pipeline_time - start_pipline_time, "seconds")

messages = [
		{"role": "system", "content": "You are a chatbot."},
		{"role": "user", "content": "How to train my dog to sit?"},
]

start_req_time = time.time()

prompt = pipeline.tokenizer.apply_chat_template(
				messages, 
				tokenize=False, 
				add_generation_prompt=True
)

terminators = [
		pipeline.tokenizer.eos_token_id,
		pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
		prompt,
		max_new_tokens=100,
		max_length=100,
		eos_token_id=terminators,
		do_sample=True,
		temperature=0.6,
		top_p=0.9,
)

end_req_time = time.time()

print("$$$$$$$$$$$$$$$$$ Response: ")
#print(json.dumps(res, indent=4))
print(outputs[0]["generated_text"][len(prompt):])
print("$$$$$$$$$$$$$$$$$ Time taken for generate response: ", end_req_time - start_req_time, "seconds")
print("$$$$$$$$$$$$$$$$$ Total Time taken: ", end_req_time - start_pipline_time, "seconds")
if hasattr(pipeline, 'device'):
		print("$$$$$$$$$$$$$$$$$ device: ", pipeline.device)
if hasattr(pipeline, 'device_map'):
		print("$$$$$$$$$$$$$$$$$ device_map: ", pipeline.device_map)
