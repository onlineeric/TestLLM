import time, os, json, torch
from transformers import pipeline

hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_time = time.time()
generator = pipeline('text2text-generation', model='lmsys/fastchat-t5-3b-v1.0',
                     token=hf_token, 
                     device=0,	# 0 for GPU, -1 for CPU
										model_kwargs={
										# "torch_dtype": torch.float16,
										# 	"quantization_config": {"load_in_4bit": True},
										"low_cpu_mem_usage": True,
    								})

end_time = time.time()
print("Time taken for pipeline: ", end_time - start_time, "seconds")

start_time = time.time()
res = generator('What is the color of an apple?', max_length=100, num_return_sequences=1)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
