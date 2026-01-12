import os
import re
import torch
import warnings
import evaluate
import json
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from peft import PeftModel

warnings.filterwarnings("ignore")

# --- Configuration ---
HF_TOKEN = ""
TXT_FILENAME = "ablation_distilgpt_results.txt"
JSON_LOG = "ablation_full_logs.json" # To save every single response
BATCH_SIZE = 16 

if HF_TOKEN:
    login(token=HF_TOKEN)

# Reference Weights for Ensemble
ENSEMBLE_WEIGHTS = [0.010, 0.010, 0.010, 0.960, 0.010]

model_configs = {
    "gemma2b": {"base": "google/gemma-2b", "adapter": "advy/gemma2-mentalchat16k"},
    "distilgpt2": {"base": "distilgpt2", "adapter": "advy/distilgpt2-mentalchat16k"},
    "phi2": {"base": "microsoft/phi-2", "adapter": "advy/phi2-mentalchat16k"},
    "mistral": {"base": "mistralai/Mistral-7B-v0.1", "adapter": "advy/mistral-mentalchat16k"},
    "tinyllama": {"base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "adapter": "advy/tinyllama-mentalchat16k"}
}
model_names = ["gemma2b", "distilgpt2", "phi2", "mistral", "tinyllama"]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- Utils ---
def clean_text(text: str) -> str:
    # Standardizes text by removing newlines and extra spaces for fair scoring
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r'\s+', ' ', text).strip()

def format_prompt(system_prompt: str, question: str, append_text: str = "") -> str:
    # Formats for MentalChat logic
    return f"<|system|>\n{system_prompt.strip()}{append_text}\n\n<|user|>\n{question}\n\nAnswer:"

def get_weighted_ref(responses, weights, emb_model):
    valid = [(r, w) for r, w in zip(responses, weights) if r.strip()]
    if not valid: return torch.zeros(384).to(device)
    resps, wts = zip(*valid)
    embs = emb_model.encode(list(resps), convert_to_tensor=True, normalize_embeddings=True)
    w = torch.tensor(wts, dtype=embs.dtype, device=embs.device).unsqueeze(1)
    return F.normalize(torch.sum(embs * w, dim=0) / torch.sum(w), p=2, dim=0)

# --- Main Logic ---
def run_ablation_study():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
    )
    pipes = {}

    # Load all Models
    for name in model_names:
        cfg = model_configs[name]
        print(f"Loading {name}...")
        tok = AutoTokenizer.from_pretrained(cfg['adapter'])
        tok.pad_token = tok.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            cfg['base'], quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, cfg['adapter'])
        pipes[name] = pipeline("text-generation", model=model, tokenizer=tok)

    distil_pipe = pipes["distilgpt2"]
    emb_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    dataset = load_dataset("ShenLab/MentalChat16K", split="train[:100]")

    log_entries = []
    
    FAIRNESS = "\n\nENSURE THE RESPONSE IS FAIR."
    DIVERSITY = "\n\nGIVE A UNIQUE ANSWER."
    ACCURACY = "\n\nBE ACCURATE."

    for i, data in enumerate(dataset):
        # The dataset instruction is the system prompt, 'input' is the user question
        sys_p = data['instruction']
        q = data['input']
        target = clean_text(data['output'])
        
        # 1. Generate Ensemble baseline
        ensemble_resps = []
        for name in model_names:
            p = format_prompt(sys_p, q, "")
            out = pipes[name](p, max_new_tokens=100, do_sample=True, temperature=0.1, return_full_text=False)
            ensemble_resps.append(clean_text(out[0]['generated_text']))
        
        ref_vec = get_weighted_ref(ensemble_resps, ENSEMBLE_WEIGHTS, emb_model)
        
        # 2. DistilGPT Ablation Candidates
        variants = ["", "", FAIRNESS, DIVERSITY, ACCURACY]
        prompts = [format_prompt(sys_p, q, v) for v in variants]
        c_raw = distil_pipe(prompts, batch_size=BATCH_SIZE, max_new_tokens=100, do_sample=True, temperature=0.7, return_full_text=False)
        candidates = [clean_text(res[0]['generated_text']) for res in c_raw]
        
        # 3. Select Best Candidate via Cosine Similarity to Ensemble
        c_embs = emb_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
        sims = F.cosine_similarity(c_embs, ref_vec.unsqueeze(0))
        best_idx = torch.argmax(sims).item()
        best_resp = candidates[best_idx]
        
        # 4. Save entry
        log_entries.append({
            "id": i + 1,
            "question": q,
            "prediction": best_resp,
            "reference": target,
            "similarity": float(torch.max(sims))
        })
        print(f"Processed Q{i+1}/100 - Sim: {sims[best_idx]:.4f}")

    # --- Final Scoring ---
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    preds = [x['prediction'] for x in log_entries]
    refs = [x['reference'] for x in log_entries]
    
    r_scores = rouge.compute(predictions=preds, references=refs)
    b_score = bleu.compute(predictions=preds, references=[[r] for r in refs])

    report = (
        f"ABLATION STUDY RESULTS - {datetime.now()}\n"
        f"{'='*40}\n"
        f"ROUGE-1: {r_scores['rouge1']:.4f}\n"
        f"ROUGE-2: {r_scores['rouge2']:.4f}\n"
        f"ROUGE-L: {r_scores['rougeL']:.4f}\n"
        f"BLEU:    {b_score['bleu']:.4f}\n"
        f"{'='*40}\n"
    )

    # Save metrics
    with open(TXT_FILENAME, "w") as f:
        f.write(report)
    
    # Save every response for manual inspection
    with open(JSON_LOG, "w") as f:
        json.dump(log_entries, f, indent=2)

    print("\n" + report)
    print(f"Full logs saved to {JSON_LOG}")

if __name__ == "__main__":
    run_ablation_study()
