from transformers import pipeline

generator = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B')
res2 = generator('What is transformer in AI?', max_length=50, num_return_sequences=5)
