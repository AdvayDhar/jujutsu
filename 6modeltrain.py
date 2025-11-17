# upgraded_mentalchat16k_finetune.py
# ==============================
# Robust multi-model QLoRA finetuning for MentalChat16K
# Adds Llama-3.1-70B-Instruct and produces README + test eval for each model.
# ==============================

import os
import json
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from typing import Dict, List, Tuple
from collections import Counter

# ----------------------------
# Configuration
# ----------------------------
HF_TOKEN = ""  # replace or set via env
os.environ["HF_TOKEN"] = HF_TOKEN

MODELS_CONFIG = {
    # NOTE: repo_name strings follow your requested naming
    "tinyllama-mental-health": {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "repo_name": "tinyllama-mentalchat16k",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_rank": 16,
        "max_length": 512,
        "batch_size": 4,
        "epochs": 4
    },
    "mistral-mental-health": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "repo_name": "mistral-mentalchat16k",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_rank": 32,
        "max_length": 768,
        "batch_size": 2,
        "epochs": 4
    },
    "phi2-mental-health": {
        "base_model": "microsoft/phi-2",
        "repo_name": "phi2-mentalchat16k",
        "target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
        "lora_rank": 16,
        "max_length": 512,
        "batch_size": 4,
        "epochs": 4
    },
    "distilgpt2-mental-health": {
        "base_model": "distilgpt2",  # corrected to the HF transformer id
        "repo_name": "distilgpt2-mentalchat16k",
        "target_modules": ["c_attn", "c_proj"],
        "lora_rank": 16,
        "max_length": 512,
        "batch_size": 4,
        "epochs": 4
    },
    "gemma2b-mental-health": {
        "base_model": "google/gemma-2b",
        "repo_name": "gemma2-mentalchat16k",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_rank": 24,
        "max_length": 640,
        "batch_size": 3,
        "epochs": 4
    },
    # New large model - Llama 3.1 70B Instruct
    "llama71b-mental-health": {
        "base_model": "meta-llama/Llama-3.1-70B-Instruct",
        "repo_name": "llama71b-mentalchat16k",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_rank": 64,   # moderate LoRA rank for 70B
        "max_length": 2048,
        "batch_size": 1,
        "epochs": 3
    }
}

# ----------------------------
# Dataset processing and validation
# ----------------------------
def validate_conversation(input_text: str, output_text: str) -> Tuple[bool, str]:
    if not input_text or input_text.isspace():
        return False, "Empty input"
    if len(input_text) < 10 or len(output_text) < 20:
        return False, "Too short"
    if len(input_text) > 1200 or len(output_text) > 2000:
        return False, "Too long"
    input_words = input_text.split()
    output_words = output_text.split()
    if len(input_words) < 3 or len(output_words) < 8:
        return False, "Not enough words"
    if len(output_words) > 0:
        from collections import Counter
        most_common = Counter(output_words).most_common(1)[0][1]
        if most_common / len(output_words) > 0.25:
            return False, "Too repetitive"
    if output_text.strip().lower() in ["ok", "yes", "no", "sure", "okay"]:
        return False, "Too simple response"
    return True, "Valid"

def process_mentalchat_dataset() -> Dataset:
    print("=== Loading MentalChat16K Dataset ===")
    dataset = load_dataset("ShenLab/MentalChat16K", split="train")
    print(f"Loaded {len(dataset):,} examples")
    SYSTEM_PROMPT = dataset[0]['instruction'].strip()
    conversations = []
    validation_stats = Counter()
    for ex in dataset:
        input_text = (ex.get('input') or "").strip()
        output_text = (ex.get('output') or "").strip()
        ok, reason = validate_conversation(input_text, output_text)
        validation_stats[reason] += 1
        if ok:
            formatted = f"""<|system|>\n{SYSTEM_PROMPT}\n\n<|user|>\n{input_text}\n\n<|assistant|>\n{output_text}"""
            conversations.append({"text": formatted})
    print("Validation stats:", validation_stats.most_common())
    processed = Dataset.from_list(conversations)
    return processed

def prepare_dataset():
    processed = process_mentalchat_dataset()
    print("Total valid examples:", len(processed))
    train_val = processed.train_test_split(test_size=0.15, seed=42, shuffle=True)
    train_ds = train_val["train"]
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)
    val_ds = val_test["train"]
    test_ds = val_test["test"]
    print("Split -> train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))
    return train_ds, val_ds, test_ds

# ----------------------------
# Model & tokenizer setup (with QLoRA / 4-bit bitsandbytes)
# ----------------------------
def setup_model_and_tokenizer(model_name: str, target_modules: List[str], lora_rank: int = 16, enable_gradient_checkpointing: bool = True):
    print(f"Setting up model: {model_name} (LoRA r={lora_rank})")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    # Try to enable gradient checkpointing for big models (reduces mem, slows compute)
    try:
        if enable_gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        # Some HF models require config changes to disable cache
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass
    # LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=max(32, lora_rank * 2),
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    try:
        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    except Exception:
        pass
    return model, tokenizer

# ----------------------------
# README generator
# ----------------------------
def generate_readme(output_dir: str, metrics: dict):
    readme_path = os.path.join(output_dir, "README.md")
    model = metrics.get("model", "model")
    base_model = metrics.get("base_model", "base_model")
    eval_loss = metrics.get("evaluation", {}).get("eval_loss", None)
    test_loss = metrics.get("test_eval", {}).get("eval_loss", None)
    text = f"""# {model}

**Base model:** {base_model}

**Fine-tuned on:** ShenLab/MentalChat16K (MentalChat16K) — 16,000 mental-health conversational examples.

## Overview
This model was fine-tuned using LoRA adapters (QLoRA-style 4-bit quantized base model) on the MentalChat16K dataset to improve performance for mental-health conversational responses.

## Training summary
- LoRA config: {json.dumps(metrics.get('lora_config', {}), indent=2)}
- Training: {json.dumps(metrics.get('training', {}), indent=2)}

## Evaluation
- Validation eval_loss: {eval_loss}
- Test eval_loss: {test_loss}

## Usage
Load the model via the Hugging Face Hub and use standard causal LM generation code. Tokenizer and model are saved in the repo.

## Notes
- Dataset used: ShenLab/MentalChat16K (MentalChat16K)
- This model is intended for research and non-emergency mental health assistance. Not a substitute for professional care.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)
    return readme_path

# ----------------------------
# Training + evaluation function
# ----------------------------
def train_model(model_config: Dict, train_dataset, val_dataset, test_dataset, model_key: str):
    print("="*60)
    print(f"Training: {model_key}")
    print("="*60)
    base_model = model_config["base_model"]
    repo_name = model_config["repo_name"]
    target_modules = model_config["target_modules"]
    lora_rank = model_config["lora_rank"]
    max_length = model_config["max_length"]
    per_device_batch = model_config["batch_size"]
    epochs = model_config.get("epochs", 3)

    try:
        model, tokenizer = setup_model_and_tokenizer(base_model, target_modules, lora_rank, enable_gradient_checkpointing=True)

        # Tokenization function
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length)

        train_tok = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names, desc="Tokenize train")
        val_tok = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names, desc="Tokenize val")
        test_tok = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names, desc="Tokenize test")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # compute gradient accumulation so effective batch is at least 8 (unless user configured differently)
        target_effective_batch = 8
        grad_accum = max(1, target_effective_batch // per_device_batch)

        # compute steps
        steps_per_epoch = max(1, len(train_tok) // (per_device_batch * grad_accum))
        total_steps = steps_per_epoch * epochs

        # LR schedule / warmup: smaller initial LR for very large models
        base_lr = 2e-4
        if "llama71b" in repo_name.lower() or "70" in base_model.lower():
            base_lr = 8e-5  # safer for 70B
        warmup_ratio = 0.03

        output_dir = f"./results/{repo_name}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=base_lr,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            eval_strategy="steps",
            eval_steps=max(100, steps_per_epoch // 10),
            save_strategy="steps",
            save_steps=max(200, steps_per_epoch // 5),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            logging_first_step=True,
            report_to="none",
            fp16=True,
            push_to_hub=True,
            hub_model_id=repo_name,
            hub_strategy="checkpoint",
            dataloader_num_workers=2,
            remove_unused_columns=False
        )

        early_stopping = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )

        # Train
        train_result = trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Validation eval (best model is loaded automatically)
        eval_results = trainer.evaluate(eval_dataset=val_tok)

        # Test evaluation
        test_results = trainer.evaluate(eval_dataset=test_tok)

        # Metrics & metadata
        metrics = {
            "model": model_key,
            "base_model": base_model,
            "dataset": "ShenLab/MentalChat16K",
            "lora_config": {"rank": lora_rank, "alpha": lora_rank * 2, "target_modules": target_modules, "dropout": 0.1},
            "training": {
                "final_train_loss": getattr(train_result, "training_loss", None),
                "total_steps": getattr(train_result, "global_step", None),
                "epochs": epochs,
                "learning_rate": base_lr,
                "per_device_batch_size": per_device_batch,
                "gradient_accumulation": grad_accum
            },
            "evaluation": eval_results,
            "test_eval": test_results,
            "dataset_stats": {"train_size": len(train_dataset), "val_size": len(val_dataset), "test_size": len(test_dataset)}
        }

        # write metrics
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate README (includes dataset mention + eval)
        generate_readme(output_dir, metrics)

        # Push to Hub (README.md + everything in output_dir will be pushed)
        trainer.push_to_hub(commit_message=f"Finetune on MentalChat16K - eval_loss: {metrics['evaluation'].get('eval_loss'):.4f}")

        # cleanup
        del model, trainer
        torch.cuda.empty_cache()
        return True, metrics

    except Exception as e:
        import traceback
        print("Error:", e)
        print(traceback.format_exc())
        torch.cuda.empty_cache()
        return False, None

# ----------------------------
# Main
# ----------------------------
def main():
    print("MentalChat16K multi-model finetune - upgraded pipeline")
    login(token=HF_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds, val_ds, test_ds = prepare_dataset()

    # Show a sample
    if len(train_ds) > 0:
        print("Sample:", train_ds[0]["text"][:300])

    results = {}
    for model_key, cfg in MODELS_CONFIG.items():
        ok, metrics = train_model(cfg, train_ds, val_ds, test_ds, model_key)
        results[cfg["repo_name"]] = {"success": ok, "metrics": metrics}
        print("-"*60)

    # Save combined results
    os.makedirs("./resultsshenlab", exist_ok=True)
    with open("./resultsshenlab/mentalchat16k_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Complete. Results saved to ./resultsshenlabs/")

if __name__ == "__main__":
    main()
