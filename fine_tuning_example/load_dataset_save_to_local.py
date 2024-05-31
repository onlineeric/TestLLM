from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("CodeKapital/CookingRecipes")

# Specify the path to save the dataset
save_path = "./cooking_recipes"

# Save the dataset to local files
dataset.save_to_disk(save_path)

print(f"Dataset saved to {save_path}")
