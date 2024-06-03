def get_formatting_prompts_func(eos_token, instruction_key="instruction", input_key="input", output_key="output"):
	def formatting_prompts_func(examples):
		alpaca_prompt = """
		### Instruction:
		{}

		### Input:
		{}

		### Response:
		{}"""

		instructions = examples[instruction_key]
		inputs       = examples[input_key]
		outputs      = examples[output_key]
		texts = []
		for instruction, input, output in zip(instructions, inputs, outputs):
			# Must add EOS_TOKEN (tokenizer.eos_token), otherwise your generation will go on forever!
			text = alpaca_prompt.format(instruction, input, output) + eos_token
			texts.append(text)
		print(f"$$$ formatting_prompts_func result: {texts}")
		return { "text" : texts, }
	return formatting_prompts_func
