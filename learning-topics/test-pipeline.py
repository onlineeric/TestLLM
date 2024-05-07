import time, os, torch
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2', device=0)
start_time = time.time()
res = generator('Explain why 0.2 + 0.3 not equals to 0.5?', max_length=200, num_return_sequences=3)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(res)
