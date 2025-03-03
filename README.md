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

### Software Requirements

- **Python**: 3.8+
- **PyTorch**: With CUDA support
- **NVIDIA Drivers**: ≥535
- **CUDA Toolkit**: ≥11.8

---

## Installation 🚀

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/ai-recipe-generator.git
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