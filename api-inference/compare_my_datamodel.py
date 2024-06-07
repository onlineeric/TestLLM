import os
import requests
from pprint import pprint

MY_API_URL = "https://api-inference.huggingface.co/models/onlineeric/pythia-160m_ft_CookingRecipes"
ORG_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/pythia-160m"

API_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
headers = {"Authorization": f"Bearer {API_TOKEN}"}
params = {
	"min_length": 300, 
	"max_length": 500,
	#"temperature": 0.5,
	}
options = {"wait_for_model": True}

def query(url, payload):
	json_payload = {"inputs": payload, 
				 #"parameters": params, 
				 "options": options}
	response = requests.post(url, headers=headers, json=json_payload)
	return response

prompt_text = "Tell me how to make Quick Barbecue Wings"
response = query(ORG_API_URL, prompt_text)
print("\n$$$ original data model response")
pprint(response.json())
response = query(MY_API_URL, prompt_text)
print("\n$$$ my data model response")
pprint(response.json())
