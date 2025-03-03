# AI Recipe Generator 🍳

An AI-powered recipe generator that creates cooking recipes from a list of ingredients using fine-tuned language models. Perfect for home cooks, meal planners, and food enthusiasts!

---

## Features ✨

- 🍳 **Generate recipes** from specified ingredients
- ⚡ **GPU-accelerated** for fast performance
- 📄 **Structured JSON output** for easy parsing
- ⏱️ **Customizable cooking time** and difficulty levels
- 📊 **Performance tracking** with detailed logs
- 🧠 **Fine-tuned model** for recipe-specific outputs

---

## Requirements 🛠️

### Hardware Requirements

**Minimum Recommended:**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 2070/3060 or equivalent)
- **RAM**: 16GB System RAM
- **Storage**: 5GB Free Space

**Tested Configuration:**
- **GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU (8GB VRAM)
- **Driver**: 571.96
- **CUDA**: 12.8
- **RAM**: 16GB DDR5
- **OS**: Windows 11

*Note: CPU-only operation is supported but significantly slower.*

---

### My Software Requirements

- **Python**: 3.11.6 (3.8+)
- **PyTorch**: With CUDA support
- **NVIDIA Drivers**: ≥535
- **CUDA Toolkit**: ≥11.8

---

## Model Information 🤖

This project supports two model configurations:

### Base Model
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Description**: A pre-trained language model optimized for natural language understanding and generation tasks, including recipe generation.
- **Usage**: Run `python generate_recipe.py` for direct recipe generation.

### Fine-tuned Model
- **Model**: Custom fine-tuned version of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Description**: A specialized model fine-tuned on recipe-specific data for enhanced performance in generating structured recipes.
- **Usage**: Run `python generate_recipe_from_fine_tune_model.py` for fine-tuned recipe generation.
- **Training Data**: Fine-tuned on 20% of the [Food.com dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), ensuring high-quality recipe outputs.
- **Memory Efficient**: Optimized for systems with limited computational resources (e.g., 8GB VRAM GPUs).
- **Features**:
  - **Structured JSON Output**: Returns recipes in a consistent, machine-readable format.
  - **Recipe-Specific Optimization**: Enhanced for generating detailed and accurate recipes.
  - **Customizable Parameters**: Supports setting cooking time, difficulty, and serving size.
  - **Strict Ingredient Usage**: Uses only provided ingredients and common seasonings (e.g., salt, pepper, vinegar).

### Example JSON Output
```json
{
  "think": "I decided to make a simple and delicious soup using the provided ingredients...",
  "clean_answer": {
    "dish_name": "Tomato-Bean Soup",
    "ingredients": ["sun-dried tomatoes", "black beans", "cannellini beans", "scallion", "olive oil", "balsamic vinegar", "salt", "pepper"],
    "seasonings": ["salt", "pepper"],
    "instructions": [
      "1. Heat olive oil in a large pot...",
      "2. Add sun-dried tomatoes and cook until softened...",
      "3. Stir in black beans and cannellini beans..."
    ],
    "cooking_time": "30 minutes",
    "difficulty": "Easy"
  }
}
```

*Note: The fine-tuned model requires additional storage space but offers more consistent recipe-focused outputs.*

---

## Installation 🚀

1. **Clone the Repository**
    ```bash
    git clone https://github.com/halojybyejy/ai-recipe-generator.git
    cd ai-recipe-generator
    ```

2. **Download Dataset (If needed)**
    ```bash
    # Download from Kaggle
    # First, install the Kaggle CLI if you haven't:
    pip install kaggle

    # Place your Kaggle API token in ~/.kaggle/kaggle.json
    # Then run:
    kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions -p ./dataset
    unzip "./dataset/food-com-recipes-and-user-interactions.zip" -d "./dataset/food-com-recipes-and-user-interactions"
    rm "./dataset/food-com-recipes-and-user-interactions.zip"
    ```

    *Note: If you don't have a Kaggle account, you can manually download the dataset from [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place it in the `./dataset/food-com-recipes-and-user-interactions` directory.*

2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Verify CUDA Installation**
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    # Should output 'True'
    ```

5. **Run the Recipe Generator**
    ```bash
    # To use the base model
    python generate_recipe.py

    # To use the fine-tuned model
    python generate_recipe_from_fine_tune_model.py
    ```

    *Note: The fine-tuned model will only work if you have previously trained and saved a model at `./fine_tuned_model`*

