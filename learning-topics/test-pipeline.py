import time, json
from transformers import pipeline

type = 'text-generation'
#type = 'text2text-generation'

generator = pipeline(type, model='gpt2', device=0)
start_time = time.time()
res = generator('The color of an apple is?', max_length=50, num_return_sequences=1)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
