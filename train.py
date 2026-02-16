# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_iris(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Fine-tuning IRIS pour l'analyse...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('json', data_files='data/train_analytics.jsonl', split='train')

    def tokenize_function(examples):
        prompts = [f"{i}
Prediction: {r}" for i, r in zip(examples['instruction'], examples['response'])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    args = TrainingArguments(output_dir="./iris-output", per_device_train_batch_size=8, num_train_epochs=3)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    trainer.train()
    model.save_pretrained("./fine_tuned_iris")

if __name__ == "__main__":
    print("Script d'entraînement IRIS prêt.")
