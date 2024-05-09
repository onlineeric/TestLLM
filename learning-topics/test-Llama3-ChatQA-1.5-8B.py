import time, os, json, torch
from transformers import pipeline

# hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_time = time.time()
generator = pipeline('text-generation', model='nvidia/Llama3-ChatQA-1.5-8B',
                     # token=hf_token, 
                     device=-1,
										model_kwargs={
										# "torch_dtype": torch.float16,
										# 	"quantization_config": {"load_in_4bit": True},
										"low_cpu_mem_usage": True,
    								})

end_time = time.time()
print("Time taken for pipeline: ", end_time - start_time, "seconds")

start_time = time.time()
res = generator('The color of an apple is', max_length=100, num_return_sequences=1)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
