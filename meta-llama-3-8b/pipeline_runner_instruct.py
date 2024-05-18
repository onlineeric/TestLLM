import time

def run_pipeline(pipeline, messages, max_new_tokens=200):
	start_time = time.time()

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
		max_new_tokens=max_new_tokens,
		#max_length=200,
		eos_token_id=terminators,
		do_sample=True,
		temperature=0.6,
		top_p=0.9,
	)

	end_time = time.time()

	if hasattr(pipeline, 'device'):
		print("$$$$$$$$$$$$$$$$$ device: ", pipeline.device)
	if hasattr(pipeline, 'device_map'):
		print("$$$$$$$$$$$$$$$$$ device_map: ", pipeline.device_map)
	print("$$$$$$$$$$$$$$$$$ Response: ")
	print(outputs[0]["generated_text"][len(prompt):])
	print("$$$$$$$$$$$$$$$$$ Total Time taken: ", end_time - start_time, "seconds")
