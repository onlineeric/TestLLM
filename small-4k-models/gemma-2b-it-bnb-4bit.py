import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "unsloth/gemma-2b-it-bnb-4bit"

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="cuda:0", 
		torch_dtype="auto", 
		trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
		{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
		{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
		{"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
# messages = [
# 		{"role": "user", "content": "Hi, chatbot. How are you today?"},
# 		{"role": "assistant", "content": "You are a chatbot. I'm good to chat with you. How can I help you today?"},
# 		{"role": "user", "content": "Tell me how to cook Ramen noodles?"},
# ]

pipe = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
)

generation_args = {
		"max_new_tokens": 1000,
		"return_full_text": False,
		"temperature": 0.0,
		"do_sample": False,	# False if temperature is 0.0
}

start_time = time.time()
outputs = pipe(messages, **generation_args)
end_time = time.time()

if hasattr(pipe, 'device'):
	print("$$$$$$ device: ", pipe.device)
if hasattr(pipe, 'device_map'):
	print("$$$$$$ device_map: ", pipe.device_map)
print("$$$$$$ Response: ")
print(outputs[0]["generated_text"])
print("$$$$$$ Total Time taken: ", end_time - start_time, "seconds")
