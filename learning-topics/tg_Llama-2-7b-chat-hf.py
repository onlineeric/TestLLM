import time, os, json, torch
from transformers import pipeline

hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_time = time.time()
generator = pipeline('text-generation', model='meta-llama/Llama-2-7b-chat-hf', 
                     token=hf_token, 
                     device=0, # 0 for GPU, -1 for CPU
										model_kwargs={
										"torch_dtype": torch.float16,
										# "quantization_config": {"load_in_8bit": True},
										# "low_cpu_mem_usage": True,
    								# "use_mem_optimized_native_fp16": True
            				})

end_time = time.time()
print("Time taken for pipeline: ", end_time - start_time, "seconds")

start_time = time.time()
res = generator('Tell me how to train my dog to sit?', max_length=100, num_return_sequences=3)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
