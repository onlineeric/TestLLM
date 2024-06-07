import os
import requests
from pprint import pprint

API_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/onlineeric/pythia-160m_ft_CookingRecipes"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
	json_payload = {"inputs": payload}
	response = requests.post(API_URL, headers=headers, json=json_payload)
	return response

response = query("How to make Fresh Strawberry Pie")
pprint(response.json())
