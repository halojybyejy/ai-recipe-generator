from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
from datetime import datetime
import os
import re

# Record Start Time
start_time = time.time()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model name
model_name = "./fine_tuned_model"

# Set JSON file name
JSON_FILE = "./records/generate_recipe_from_fine_tune_model.json"

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0}  # Force all layers to GPU 0
)

# Generate prompt function
def generate_prompt(ingredients, difficulty, number, people, cooking_time):
    prompt = f"""
        You are a chef. Create a recipe using **only** the provided ingredients and **common seasonings** (e.g., salt, pepper, vinegar, soy sauce). 
        Do **not** add any extra ingredients.

        - **Ingredients**: {', '.join(ingredients)}
        - **Difficulty**: {difficulty}
        - **Cooking time**: {cooking_time}
        - **Servings**: {people} people
        - **Dishes**: {number}

        Your response **must be in valid JSON format** with the following structure:

        ```json
        {{
            "think": "Your internal thoughts about how to create this recipe.",
            "clean_answer": {{
                "dish_name": "Dish Name Here",
                "ingredients": ["List of Ingredients"],
                "seasonings": ["List of Seasonings"],
                "instructions": [
                    "Step 1",
                    "Step 2",
                    "..."
                ],
                "cooking_time": "{cooking_time}",
                "difficulty": "{difficulty}"
            }}
        }}
        ```"""
    return prompt

# Generate answer function
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=3000,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Parse response function
def parse_response(response):
    """Returns: (status, error_message, parsed_data)"""
    try:
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        for json_str in reversed(json_blocks):
            try:
                json_str = json_str.strip()
                # Correct trailing comma regex
                json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                # Only fix unquoted keys if necessary
                json_str = re.sub(r'([{,])(\s*)(\w+)(\s*):', r'\1"\3":', json_str)
                # Handle unquoted string values carefully
                json_str = re.sub(r':\s*([a-zA-Z_][^,}\n"]*)', r': "\1"', json_str)
                
                data = json.loads(json_str)
                if "clean_answer" in data:
                    return ("success", None, {
                        "think": data.get("think", ""),
                        "clean_answer": data["clean_answer"]
                    })
                elif "dish_name" in data:
                    return ("success", None, {"think": "", "clean_answer": data})
            except json.JSONDecodeError as e:
                continue

        # Fallback logic (if needed)
        # ... (keep existing fallback code)

        return ("error", "No valid recipe structure found", None)
    except Exception as e:
        return ("error", f"Unexpected error: {str(e)}", None)
    
# Load JSON function
def load_json():
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print("⚠️ Warning: JSON file is empty or corrupted. Creating a new one.")
            return []
    return []

# Save to JSON function
def save_to_json(prompt, response, start_time, end_time):
    """Save results with detailed error handling"""
    data = load_json()
    total_time = round(end_time - start_time, 2)
    
    # Parse response
    status, error, parsed_data = parse_response(response)
    
    entry = {
        "metadata": {
            "status": status,
            "timestamp": datetime.fromtimestamp(end_time).isoformat(),
            "processing_time": total_time,
            "error": error
        },
        "prompt": prompt,
        "raw_response": response,
        "think_response": parsed_data.get("think") if status == "success" else None,
        "clean_response": parsed_data.get("clean_answer") if status == "success" else None
    }
    
    data.append(entry)
    
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved with status: {status.upper()}" + (f" - {error}" if error else ""))

# Main execution
if __name__ == "__main__":
    # Set parameters
    ingredients = ['dark roast coffee', 'ground cardamom', 'water', 'almond milk', 'sugar', 'chocolate syrup']
    difficulty = "Easy"
    number = 1
    cooking_time = "30 minutes"
    people = 3

    # Generate prompt
    prompt = generate_prompt(ingredients, difficulty, number, people, cooking_time)
    
    # Generate answer
    response = generate_answer(prompt)
    
    # Parse response
    status, error_info, parsed_data = parse_response(response)  # Fixed unpacking
    
    # Record end time
    end_time = time.time()
    
    # Save to JSON
    save_to_json(prompt, response, start_time, end_time)
    
    # Print results
    print("\n=== Processing Results ===")
    print(f"Status: {status.upper()}")
    if status == "error":
        print(f"Error: {error_info}")
    else:
        print("\nParsed Clean Answer:")
        print(json.dumps(parsed_data['clean_answer'], indent=4))
    print("\nGenerated Response:")
    print(response)