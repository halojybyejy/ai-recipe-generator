import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import ast
import numpy as np
import pandas as pd

# ✅ Set dataset path
DATASET_PATH = "./dataset/food-com-recipes-and-user-interactions/versions/2/RAW_recipes.csv"

# ✅ Load the dataset into a DataFrame
df = pd.read_csv(DATASET_PATH)

# ✅ Randomly sample 1% (or adjust percentage)
df_sampled = df.sample(frac=0.2, random_state=42)  # 50% of data

# ✅ Save the reduced dataset
SMALL_DATASET_PATH = "C:/projects/generate_recipe/dataset/food-com-recipes-and-user-interactions/versions/2/SMALL_recipes.csv"
df_sampled.to_csv(SMALL_DATASET_PATH, index=False)

print(f"✅ Reduced dataset saved to: {SMALL_DATASET_PATH}")

# ✅ Load tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 computation
    bnb_4bit_use_double_quant=True  # Use double quantization for better performance
)

# ✅ Preprocess function
def preprocess_function(examples):
    text = f"Recipe: {examples['name']}\n"
    ingredients = ", ".join(ast.literal_eval(examples['ingredients'])) if isinstance(examples['ingredients'], str) else examples['ingredients']
    text += f"Ingredients: {ingredients}\n"
    steps = " ".join(ast.literal_eval(examples['steps'])) if isinstance(examples['steps'], str) else examples['steps']
    text += f"Instructions: {steps}\n"

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    return tokenized

# ✅ Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss_func(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1), 
            ignore_index=tokenizer.pad_token_id
        )
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    print("✅ Loading dataset...")
    dataset = load_dataset("csv", data_files=SMALL_DATASET_PATH)  # use DATASET_PATH for full dataset

    print("✅ Dataset loaded successfully!")
    print("Sample data:", dataset["train"][0])

    dataset = dataset["train"].train_test_split(test_size=0.2)

    # ✅ Load model with FP16 or BF16
    print("🚀 Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},  # Auto-assign layers
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # ✅ Use BF16 if possible
        quantization_config=quantization_config 
    )

    # ✅ Ensure model is fully loaded before training
    if torch.cuda.is_available():
        model = model.to("cuda")

    print("✅ Model loaded successfully!")

    # ✅ Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # ✅ Add LoRA to the model
    model = get_peft_model(model, lora_config)

    print("✅ Tokenizing dataset...")

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )

    print("✅ Tokenization complete!")
    print("Tokenized dataset sample:", tokenized_datasets["train"][0])

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=True  # ✅ Enable FP16 training
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ✅ Start Training
    print("🚀 Training started...")
    trainer.train()
    print("🎉 Training completed!")

    # ✅ Save fine-tuned model
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("✅ Fine-tuned model saved at `./fine_tuned_model`")
