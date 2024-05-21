from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda:0")

messages = [
		{"role": "user", "content": "what is the percentage change of the net income from Q4 FY23 to Q4 FY24?"}
]

document = """NVIDIA (NASDAQ: NVDA) today reported revenue for the fourth quarter ended January 28, 2024, of $22.1 billion, up 22% from the previous quarter and up 265% from a year ago.\nFor the quarter, GAAP earnings per diluted share was $4.93, up 33% from the previous quarter and up 765% from a year ago. Non-GAAP earnings per diluted share was $5.16, up 28% from the previous quarter and up 486% from a year ago.\nQ4 Fiscal 2024 Summary\nGAAP\n| $ in millions, except earnings per share | Q4 FY24 | Q3 FY24 | Q4 FY23 | Q/Q | Y/Y |\n| Revenue | $22,103 | $18,120 | $6,051 | Up 22% | Up 265% |\n| Gross margin | 76.0% | 74.0% | 63.3% | Up 2.0 pts | Up 12.7 pts |\n| Operating expenses | $3,176 | $2,983 | $2,576 | Up 6% | Up 23% |\n| Operating income | $13,615 | $10,417 | $1,257 | Up 31% | Up 983% |\n| Net income | $12,285 | $9,243 | $1,414 | Up 33% | Up 769% |\n| Diluted earnings per share | $4.93 | $3.71 | $0.57 | Up 33% | Up 765% |"""

def get_formatted_input(messages, context):
		system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
		instruction = "Please give a full and complete answer for the question."

		for item in messages:
				if item['role'] == "user":
						## only apply this instruction for the first user turn
						item['content'] = instruction + " " + item['content']
						break

		conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
		formatted_input = system + "\n\n" + context + "\n\n" + conversation
		print('$$$$$$ formatted_input:', formatted_input);
		
		return formatted_input

formatted_input = get_formatted_input(messages, document)

start_time = time.time()
tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
end_time = time.time()
print("$$$$$$ tokenized prompt: ", end_time - start_time, "seconds")

terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

start_time = time.time()
outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=1000, eos_token_id=terminators)
end_time = time.time()

response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
if hasattr(model, 'device'):
	print("$$$$$$ device: ", model.device)
if hasattr(model, 'device_map'):
	print("$$$$$$ device_map: ", model.device_map)
print("$$$$$$ Response: ")
print(tokenizer.decode(response, skip_special_tokens=True))
print("$$$$$$ Total Time taken: ", end_time - start_time, "seconds")
