import time, json
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2', device=0)
start_time = time.time()
res = generator('The color of an apple is', max_length=100, num_return_sequences=3)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
