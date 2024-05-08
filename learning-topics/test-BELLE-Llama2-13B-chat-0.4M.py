import time, json
from transformers import pipeline

generator = pipeline('text2text-generation', model='BELLE-2/BELLE-Llama2-13B-chat-0.4M', device=0)
start_time = time.time()
res = generator('Why my apple\s color is not in red?', max_length=50, num_return_sequences=3)
end_time = time.time()
print("Time taken: ", end_time - start_time, "seconds")
print(json.dumps(res, indent=4))
