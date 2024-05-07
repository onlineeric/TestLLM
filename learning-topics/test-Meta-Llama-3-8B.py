import time, os, torch
from transformers import pipeline

hf_token = os.getenv('HUGGINGFACE_TOKEN')

generator = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B', 
                     token=hf_token, 
                     #device=0,
										 model_kwargs={
											"torch_dtype": torch.float16,
											"quantization_config": {"load_in_4bit": True},
											"low_cpu_mem_usage": True,
    								})
start_time = time.time()
res = generator('Explain why 0.2 + 0.3 not equals to 0.5?', max_length=200, num_return_sequences=3)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(res)
