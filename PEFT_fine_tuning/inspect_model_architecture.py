from transformers import AutoModelForCausalLM

model_name = "EleutherAI/pythia-160m"
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
