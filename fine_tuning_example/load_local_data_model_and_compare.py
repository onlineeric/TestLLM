from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import inference

model_id = "EleutherAI/pythia-410m"
finetuned_model_dir = "./gitignore_trained_models/pythia-410m-finetuned-cooking_recipes/final"

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(finetuned_model_dir, local_files_only=True)
finetuned_slightly_model.to('cuda')
finetuned_slightly_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir, local_files_only=True)

hf_model = AutoModelForCausalLM.from_pretrained(model_id)
hf_model.to('cuda')
hf_tokenizer = AutoTokenizer.from_pretrained(model_id)

test_dataset = [
	"Wash potatoes; prick several times with a fork. Microwave them with a wet paper towel covering the potatoes on high for 6-8 minutes.",
	"In a slow cooker, combine all ingredients. Cover and cook on low for 4 hours or until heated through and cheese is melted. Stir well before serving. Yields 6 servings.",
	"Boil and debone chicken.", "Put bite size pieces in average size square casserole dish.", "Pour gravy and cream of mushroom soup over chicken; level.", "Make stuffing according to instructions on box (do not make too moist).", "Put stuffing on top of chicken and gravy; level.", "Sprinkle shredded cheese on top and bake at 350\u00b0 for approximately 20 minutes or until golden and bubbly.",
	"Wash potatoes; prick several times with a fork.", "Microwave them with a wet paper towel covering the potatoes on high for 6-8 minutes.", "The potatoes should be soft, ready to eat.", "Let them cool enough to handle.", "Cut in half lengthwise; scoop out pulp and reserve.", "Discard shells.", "Brown ground beef until done.", "Drain any grease from the meat.", "Set aside when done.", "Meat will be added later.", "Melt butter in a large kettle over low heat; add flour, stirring until smooth.", "Cook 1 minute, stirring constantly. Gradually add milk; cook over medium heat, stirring constantly, until thickened and bubbly.", "Stir in potato, ground beef, salt, pepper, 1 cup of cheese, 2 tablespoons of green onion and 1/2 cup of bacon.", "Cook until heated (do not boil).", "Stir in sour cream if desired; cook until heated (do not boil).", "Sprinkle with remaining cheese, bacon and green onions.",
	"Roll steak strips in flour.", "Brown in skillet.", "Salt and pepper.", "Combine tomato liquid, water, onions and browned steak. Cover and simmer for one and a quarter hours.", "Uncover and stir in Worcestershire sauce.", "Add tomatoes, green peppers and simmer for 5 minutes.", "Serve over hot cooked rice.",
	"Drain pears, reserving juice.", "Bring juice to a boil, stirring constantly.", "Remove from heat.", "Add gelatin, stirring until dissolved.", "Let cool slightly.", "Coarsely chop pear halves. Combine cream cheese and yogurt; beat at medium speed of electric mixer until smooth.", "Add gelatin and beat well.", "Stir in pears.", "Pour into an oiled 4-cup mold or Pyrex dish.", "Chill.",
]

for i in range(1):
	test_question = test_dataset[i]
	print("\n$$$ Question input (test):", test_question)

	print("$$$ Hugging Face model's answer: ")
	print(inference(test_question, hf_model, hf_tokenizer, 1000, 200))
 
	print("$$$ Finetuned slightly model's answer: ")
	print(inference(test_question, finetuned_slightly_model, finetuned_slightly_tokenizer, 1000, 200))
