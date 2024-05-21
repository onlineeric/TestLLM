from pprint import pprint
from transformers import AutoTokenizer

model_id = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Hi, how are you?"

# encode and decode text
encoded_text = tokenizer(text)["input_ids"]
print(encoded_text)
decoded_text = tokenizer.decode(encoded_text)
print("Decoded tokens back into text: ", decoded_text)

# encode and decode list of texts
list_texts = ["Hi, how are you?", "I'm good", "Yes"]
encoded_texts = tokenizer(list_texts)
print("Encoded list of texts: ", encoded_texts["input_ids"])

# Padding
tokenizer.pad_token = tokenizer.eos_token 
encoded_texts_longest = tokenizer(list_texts, padding=True)
print("Using padding: ", encoded_texts_longest["input_ids"])

# Truncation
encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
print("Using truncation: ", encoded_texts_truncation["input_ids"])

# Truncation left side
tokenizer.truncation_side = "left"
encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
print("Using left-side truncation: ", encoded_texts_truncation_left["input_ids"])

# Padding and Truncation
tokenizer.truncation_side = "right" # reset truncation side
encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
print("Using both padding and truncation: ", encoded_texts_both["input_ids"])