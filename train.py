# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_iris(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Device détecté : {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "train_analytics.jsonl")
    dataset = load_dataset('json', data_files=data_path, split='train')

    def tokenize_function(examples):
        prompts = [f"{i}\nPrediction: {r}" for i, r in zip(examples['instruction'], examples['response'])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    args = TrainingArguments(
        output_dir="./iris-output",
        per_device_train_batch_size=16, # Dataset petit, on peut monter le batch
        num_train_epochs=3,
        use_mps_device=True if device == "mps" else False,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    print("--- Démarrage de l'entraînement IRIS ---")
    trainer.train()
    model.save_pretrained("./fine_tuned_iris")
    print("✅ IRIS entraîné et sauvegardé.")

if __name__ == "__main__":
    train_iris()
