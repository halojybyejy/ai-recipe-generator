from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    # device_map="auto"  # Let Accelerate handle device allocation
    device_map={"": 0}  # Force all layers to GPU 0
)

def generate_recipe(favorite_ingredients, favorite_cooking_methods, ingredients, meal_type, required_ingredients, difficulty, number, people, time):
    
    prompt = f"""You are a chef. Create a recipe using **only** the provided ingredients and **common seasonings** (e.g., salt, pepper, vinegar, soy sauce). Do **not** add any extra ingredients.

        - **Ingredients**: {', '.join(ingredients)}
        - **Must include**: {', '.join(required_ingredients) if required_ingredients else 'Any'}
        - **Difficulty**: {difficulty}
        - **Meal type**: {meal_type}
        - **Cooking methods**: {', '.join(favorite_cooking_methods) if favorite_cooking_methods else 'Any'}
        - **Cooking time**: {time}
        - **Servings**: {people} people
        - **Dishes**: {number}
        - **Prioritize**: {', '.join(favorite_ingredients)}

        **Only use these ingredients + seasonings. Do NOT add any other ingredients. Response length should be in 2000 words.**

        ### **Format:**
        - **Dish Name**
        - **Ingredients (Include Quantities)**
        - **Seasonings (If Used)**
        - **Instructions**
        - **Cooking Time & Difficulty**

        Provide only the recipe, no extra explanations.
        """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=3000,  # Shorter response
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


favourite_ingredients = ["Egg"]
favourite_cooking_method = ["Fry", "Soup"]
# ingredients = ["Carrot", "Potato", "Onion", "Tomato", "Egg"]
ingredients = ['pineapple', 'mangoes', 'banana', 'red peppers', 'watercress leaf']
required_ingredients = ["Potato"]
meal_type = "Dinner"
difficulty = "Easy"
number = 1
time = "30 minutes"
people = 3

recipe = generate_recipe(favourite_ingredients, favourite_cooking_method, ingredients, meal_type, required_ingredients, difficulty, number, people, time)
print(recipe)
