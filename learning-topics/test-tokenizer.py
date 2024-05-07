from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline('sentiment-analysis')
res = classifier('Hi, who are you?')
print('$$$$$$$$$ res1:', res)	# res1: [{'label': 'POSITIVE', 'score': 0.9914649128913879}]

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
res = classifier('Hi, who are you?')
print('$$$$$$$$$ res2:', res) # res2: [{'label': 'POSITIVE', 'score': 0.9914649128913879}]

sequence = "I've been waiting for a HuggingFace course the whole year!"
res = tokenizer(sequence)
print('$$$$$$$$$ tokenizer res:', res) # tokenizer res: {'input_ids': [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 1996, 2878, 2095, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
tokens = tokenizer.tokenize(sequence)
print('$$$$$$$$$ tokens:', tokens) # tokens: ['i', "'", 've', 'been', 'waiting', 'for', 'a', 'hugging', '##face', 'course', 'the', 'whole', 'year', '!']
ids = tokenizer.convert_tokens_to_ids(tokens)
print('$$$$$$$$$ ids:', ids) # ids: [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 1996, 2878, 2095, 999]
decoded = tokenizer.decode(ids)
print('$$$$$$$$$ decoded:', decoded) # decoded: i've been waiting for a huggingface course the whole year!
