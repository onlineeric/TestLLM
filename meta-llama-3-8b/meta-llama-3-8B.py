import transformers
import torch
import time
import json
import os

model_id = "meta-llama/Meta-Llama-3-8B"
hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_pipline_time = time.time()

generator = transformers.pipeline(
		"text-generation", 
		token=hf_token,
		model=model_id, 
		model_kwargs={"torch_dtype": torch.bfloat16}, 
		### below can only choose one
		#device_map="auto",
		device=0,
		#device=-1,
)

end_pipeline_time = time.time()
print("$$$$$$$$$$$$$$$$$ Time taken for pipeline: ", end_pipeline_time - start_pipline_time, "seconds")

start_req_time = time.time()
res = generator("How to train my dog to sit?")
end_req_time = time.time()

print("$$$$$$$$$$$$$$$$$ Response: ")
print(json.dumps(res, indent=4))
print("$$$$$$$$$$$$$$$$$ Time taken for generate response: ", end_req_time - start_req_time, "seconds")
print("$$$$$$$$$$$$$$$$$ Total Time taken: ", end_req_time - start_pipline_time, "seconds")

