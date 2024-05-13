import time

def run_pipeline(pipeline, messages, max_new_tokens=200):
	start_pipline_time = time.time()

	end_pipeline_time = time.time()
	print("$$$$$$$$$$$$$$$$$ Time taken for pipeline: ", end_pipeline_time - start_pipline_time, "seconds")

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
		max_new_tokens=200,
		#max_length=200,
		eos_token_id=terminators,
		do_sample=True,
		temperature=0.6,
		top_p=0.9,
	)

	end_req_time = time.time()

	if hasattr(pipeline, 'device'):
		print("$$$$$$$$$$$$$$$$$ device: ", pipeline.device)
	if hasattr(pipeline, 'device_map'):
		print("$$$$$$$$$$$$$$$$$ device_map: ", pipeline.device_map)
	print("$$$$$$$$$$$$$$$$$ Response: ")
	print(outputs[0]["generated_text"][len(prompt):])
	print("$$$$$$$$$$$$$$$$$ Time taken for generate response: ", end_req_time - start_req_time, "seconds")
	print("$$$$$$$$$$$$$$$$$ Total Time taken: ", end_req_time - start_pipline_time, "seconds")
