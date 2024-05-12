import time, os, json, torch
from transformers import pipeline

hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_pipline_time = time.time()
generator = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B', 
										 token=hf_token,
										 device=-1,
										model_kwargs={
										# "torch_dtype": torch.float16,
										# 	"quantization_config": {"load_in_4bit": True},
										"low_cpu_mem_usage": True,
										})

end_pipeline_time = time.time()
print("$$$$$$$$$$$$$$$$$ Time taken for pipeline: ", end_pipeline_time - start_pipline_time, "seconds")

start_req_time = time.time()
res = generator('Tell me how to train my dog to sit.', max_length=100, num_return_sequences=3)
end_req_time = time.time()

print("$$$$$$$$$$$$$$$$$ Response: ")
print(json.dumps(res, indent=4))
print("$$$$$$$$$$$$$$$$$ Time taken for generate response: ", end_req_time - start_req_time, "seconds")
print("$$$$$$$$$$$$$$$$$ Total Time taken: ", end_req_time - start_pipline_time, "seconds")
