import os
import random
import numpy as np
import re
import warnings
import time
from datetime import datetime
from collections import Counter
import subprocess

# PyTorch and Transformers
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import login

# Dataset and Metrics
from datasets import load_dataset, Dataset
# Git Integration (Requires: pip install gitpython)
from git import Repo, GitCommandError 

warnings.filterwarnings("ignore")

# --- Configuration ---
# !!! IMPORTANT: FILL IN YOUR TOKENS HERE !!!
# The script CANNOT run without these for model access and GitHub uploads.
GITHUB_TOKEN = "" 
HF_TOKEN = ""

# The name of the results file
TXT_FILENAME = "ensembleresults.txt" 

# Batch size for concurrent GPU inference. Set this based on your GPU memory. 
# 16 is a good starting point for 4-bit quantized models.
BATCH_SIZE = 16 

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"Warning: Failed to log in to HuggingFace. Ensure token is valid. Error: {e}")
    
# --- Alpha Configurations ---
alpha_configs = [
    [0.010, 0.010, 0.010, 0.960, 0.010], # Best Perform
    [0.010, 0.010, 0.919, 0.051, 0.010], # Best Fairness
    [0.250, 0.045, 0.226, 0.244, 0.236], # Best Diversity
    [0.010, 0.010, 0.150, 0.572, 0.258], # Balanced output
]

config_names = [
    "Best Perform",
    "Best Fairness", 
    "Best diversity",
    "Balanced output"
]

# --- Reproducibility ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Model Setup ---
models_info = {
    "gemma2b": "advy/gemma2-mentalchat16k",
    "distilgpt2": "advy/distilgpt2-mentalchat16k",
    "phi2": "advy/phi2-mentalchat16k",
    "mistral": "advy/mistral-mentalchat16k",
    "tinyllama": "advy/tinyllama-mentalchat16k"
}
model_names = ["gemma2b", "distilgpt2", "phi2", "mistral", "tinyllama"]
models_info = {name: models_info[name] for name in model_names}


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Quantization config for 4-bit
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )
else:
    bnb_config = None

def load_ensemble_models():
    """Load all ensemble models with 4-bit quantization"""
    model_pipelines = {}
    failed_models = []

    for name, mid in models_info.items():
        try:
            print(f"Loading {name} with 4-bit quantization...")
            # --- FIX 2: Set padding_side='left' for correct batch generation ---
            tok = AutoTokenizer.from_pretrained(
                mid, 
                use_fast=True, 
                trust_remote_code=False, 
                padding_side='left'
            )
            
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
            model_kwargs = {"trust_remote_code": False, "torch_dtype": dtype}
            
            if device == "cuda" and bnb_config:
                model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
            else:
                model_kwargs["device_map"] = "auto" if device == "cuda" else None
                
            model = AutoModelForCausalLM.from_pretrained(mid, **model_kwargs)
            
            # --- FIX 1: Removed `device=device` to avoid conflict with device_map="auto" ---
            pipe = pipeline("text-generation", model=model, tokenizer=tok)
            
            model_pipelines[name] = pipe
            print(f"✓ Loaded {name}")
            
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            failed_models.append(name)

    if len(model_pipelines) == 0:
        raise RuntimeError("No models loaded successfully.")

    return model_pipelines, failed_models

# --- Text Generation Utils ---

GEN_KW = dict(
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.05,
    return_full_text=False
)

# --- New Global Constants for 5-Round Prompt Variation ---
FAIRNESS_APPEND = "\n\nPLEASE ENSURE THAT THE RESPONSE IS FAIR AND DOES NOT CONTAIN OR PROMOTE DISCRIMINATION, RACISM, SEXISM OR ANY HATRED."
DIVERSITY_APPEND = "\n\nPLEASE GIVE UNIQUE AND INNOVATIVE ANSWERS"
ACCURACY_APPEND = "\n\nPLEASE MAKE THE RESPONSE AS ACCURATE AND INFORMATIVE AS POSSIBLE."

def clean_response(response: str) -> str:
    """Clean response while preserving complete content"""
    if not response or not response.strip():
        return ""
    
    response = re.sub(r'\s+', ' ', response).strip()
    
    prefixes_to_remove = [
        "Assistant:", "Answer:", "Response:", "A:", 
        "Human:", "Question:", "Q:",
        "Here is", "Here's",
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    
    response = re.sub(r'^(Here is (the|my|an) (answer|response)[\s:]*)', '', response, flags=re.IGNORECASE).strip()
    
    if response and response[-1] not in '.!?':
        last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
        if last_period > len(response) * 0.3:
            response = response[:last_period + 1]
        else:
            response = response.rstrip() + '.'
    
    return response

def format_prompt(system_prompt: str, question: str, append_text: str = "") -> str:
    """Formats the full prompt with the system instruction and question."""
    
    full_system = system_prompt.strip() + append_text
    formatted = f"""<|system|>\n{full_system}\n\n<|user|>\n{question}"""
    final_prompt = formatted + "\n\nAnswer the following question in under 100 words. Provide a complete, helpful response.\n\nAnswer:"
    
    return final_prompt

# --- NEW: Batched Generation Function ---
def generate_responses_in_batch(pipe, prompt_list: list[str]) -> list[str]:
    """Generate responses from a single model using a list of prompts for batch efficiency."""
    if not prompt_list:
        return []
    try:
        # Use the batch_size argument to maximize GPU usage
        out = pipe(
            prompt_list,
            batch_size=BATCH_SIZE,  
            **GEN_KW, 
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        responses = [clean_response(res[0]["generated_text"].strip()) for res in out]
        return responses
        
    except Exception as e:
        print(f"Error during batched generation: {e}")
        return [""] * len(prompt_list)

def weighted_average_embedding(responses, weights, embedding_model):
    """Compute weighted average of response embeddings"""
    if len(responses) != len(weights):
        raise ValueError("responses and weights must match")
    
    valid = [(r, w) for r, w in zip(responses, weights) if r.strip()]
    if not valid:
        raise ValueError("No valid responses")
    
    responses, weights = zip(*valid)
    # Sentence Transformers typically handles the batching internally, so this remains sequential per model's output
    embs = embedding_model.encode(list(responses), convert_to_tensor=True, normalize_embeddings=True)
    w = torch.tensor(weights, dtype=embs.dtype, device=embs.device).unsqueeze(1)
    weighted = torch.sum(embs * w, dim=0) / torch.sum(w)
    
    return F.normalize(weighted, p=2, dim=0)

def generate_diverse_candidates_batched(model_pipes, system_prompt: str, question: str):
    """Generates 5 candidates per model using batching."""
    all_candidates = []
    
    prompt_variations = [
        "",                     # 1. As-is
        "",                     # 2. As-is
        FAIRNESS_APPEND,        # 3. Fairness
        DIVERSITY_APPEND,       # 4. Diversity
        ACCURACY_APPEND         # 5. Accuracy
    ]
    
    for model_name, pipe in model_pipes.items():
        # 1. Collect all prompts for this model
        prompts_to_run = [format_prompt(system_prompt, question, append_text) for append_text in prompt_variations]
        
        # 2. Run generation in a single batch call
        responses = generate_responses_in_batch(pipe, prompts_to_run)
        
        # 3. Collect valid responses
        for response in responses:
            if response and response.strip() and len(response) > 20:
                all_candidates.append(response)
    
    unique_candidates = list(set(all_candidates))
    # print(f"Generated {len(unique_candidates)} unique candidates from all models/rounds.") # Keeping print optional
    return unique_candidates

def pick_best_candidate_with_config(system_prompt, question, model_pipes, weights, embedding_model):
    """Pick best candidate using the 5-round strategy and specific weight configuration."""
    
    # --- Step 1: Generate Candidates (5 rounds per model, now batched) ---
    candidates = generate_diverse_candidates_batched(model_pipes, system_prompt, question)
    
    if not candidates:
        return "", 0.0
    
    # --- Step 2: Generate Ensemble Responses (1 response per model for embedding) ---
    # Collect all standard prompts for the ensemble vector calculation
    ensemble_prompts = [format_prompt(system_prompt, question, append_text="") for _ in model_pipes]
    
    # Run inference for the ensemble responses in one giant batch across all models
    ensemble_responses_raw = []
    for pipe in model_pipes.values():
        # Note: We run a batch of size 1 (the standard prompt) for *each* model pipe here.
        # This is sequential per model but efficient for the single required output.
        # The main gain is in Step 1 where 5 prompts/model are batched together.
        try:
            resp = generate_responses_in_batch(pipe, [ensemble_prompts[0]])[0]
            ensemble_responses_raw.append(resp)
        except Exception:
            ensemble_responses_raw.append("")
        
    ensemble_responses, ensemble_weights = [], []
    for resp, w in zip(ensemble_responses_raw, weights):
        if resp and resp.strip():
            ensemble_responses.append(resp)
            ensemble_weights.append(w)
            
    if not ensemble_responses:
        print("Warning: No ensemble responses generated, returning first candidate.")
        return candidates[0], 0.0
    
    # --- Step 3: Find Best Candidate ---
    ensemble_vec = weighted_average_embedding(ensemble_responses, ensemble_weights, embedding_model)
    cand_embs = embedding_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    sims = F.cosine_similarity(cand_embs, ensemble_vec.unsqueeze(0))
    best_idx = torch.argmax(sims).item()
    
    return candidates[best_idx], sims[best_idx].item()

# --- Dataset Processing ---
def validate_conversation(input_text: str, output_text: str) -> tuple[bool, str]:
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
        most_common = Counter(output_words).most_common(1)[0][1]
        if most_common / len(output_words) > 0.25:
            return False, "Too repetitive"
    if output_text.strip().lower() in ["ok", "yes", "no", "sure", "okay"]:
        return False, "Too simple response"
    return True, "Valid"

def prepare_dataset(n_examples=100) -> Dataset:
    """Loads, validates, splits, and selects the first N examples from the test set."""
    print("Loading MentalChat16K dataset...")
    dataset = load_dataset("ShenLab/MentalChat16K", split="train")
    
    SYSTEM_PROMPT = dataset[0].get('instruction', '').strip()
    
    conversations = []
    for ex in dataset:
        input_text = (ex.get('input') or "").strip()
        output_text = (ex.get('output') or "").strip()
        ok, _ = validate_conversation(input_text, output_text)
        if ok:
            conversations.append({
                "prompt": input_text, 
                "target": output_text, 
                "system_prompt": SYSTEM_PROMPT
            })
    
    processed = Dataset.from_list(conversations)
    print(f"Total valid processed examples: {len(processed)}")

    train_val = processed.train_test_split(test_size=0.15, seed=42, shuffle=True)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)
    
    test_ds = val_test["test"].select(range(min(n_examples, len(val_test["test"]))))
    
    print(f"Test set size selected: {len(test_ds)} examples")
    return test_ds

# --- GitHub Upload Function ---
def upload_to_github(file_path: str, message: str, github_token: str):
    """
    Commits and pushes the results file to GitHub.
    Requires 'git' installed and the script running inside a cloned repository.
    """
    if not github_token:
        print("Skipping GitHub upload: GITHUB_TOKEN is empty.")
        return

    print(f"\n[GitHub] Attempting to upload '{file_path}'...")
    
    # Determine the repository root
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        if 'origin' not in repo.remotes:
            print("✗ Git failure: Repository has no remote 'origin'.")
            return
        
        if not os.path.exists(file_path):
            print(f"✗ Git failure: Results file '{file_path}' not found.")
            return

        repo.index.add([file_path])

        if not repo.index.diff("HEAD"):
             print(f"✓ No changes detected in '{file_path}'. Skipping commit.")
             return

        repo.index.commit(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

        origin = repo.remote(name='origin')
        origin.push()
        
        print(f"✓ Successfully committed and pushed checkpoint: {message}")
        
    except GitCommandError as e:
        print(f"✗ Git command failed. Check your local git setup and authentication.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"✗ Failed to interact with Git repository: {e}")

# --- Main Comparison Function ---
def run_multi_config_comparison():
    """Run comparison across multiple ensemble configurations"""
    
    def log_txt(line: str = ""):
        with open(TXT_FILENAME, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)
    
    if os.path.exists(TXT_FILENAME):
        os.remove(TXT_FILENAME)
    
    try:
        test_ds = prepare_dataset(n_examples=100)
        total_questions = len(test_ds)
    except Exception as e:
        log_txt(f"CRITICAL ERROR: Failed to load dataset: {e}")
        print(f"CRITICAL ERROR: Failed to load dataset: {e}")
        return

    # --- Header ---
    log_txt("=" * 100)
    log_txt("MULTI-CONFIGURATION ENSEMBLE COMPARISON (4-BIT QUANTIZATION)")
    log_txt("=" * 100)
    log_txt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_txt(f"Device: {device}")
    log_txt(f"Data type: {dtype}")
    log_txt(f"Quantization: 4-bit NF4")
    log_txt(f"Inference Batch Size: {BATCH_SIZE} (for candidate generation)")
    log_txt(f"Total Questions (Test Split): {total_questions}")
    log_txt(f"Generation Strategy: 5 rounds per model per question (Batched).")
    log_txt("")
    log_txt("Alpha Configurations:")
    for name, config in zip(config_names, alpha_configs):
        log_txt(f"  {name}: {config}")
    log_txt("-" * 100)

    try:
        print("Loading models with 4-bit quantization...")
        model_pipelines, failed_models = load_ensemble_models()
        
        if len(alpha_configs[0]) != len(model_pipelines):
            log_txt(f"Warning: Alpha configs adjusted to match {len(model_pipelines)} loaded models.")
            num_models = len(model_pipelines)
            alpha_configs[:] = [[1.0 / num_models] * num_models for _ in alpha_configs]

        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        # Move embedding model to GPU for speed
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        print(f"\nProcessing {total_questions} questions...")
        
        # --- Main Loop ---
        for q_idx, data in enumerate(test_ds):
            question = data['prompt']
            reference_answer = data['target']
            system_prompt = data['system_prompt']
            
            log_txt("=" * 100)
            log_txt(f"QUESTION {q_idx+1}/{total_questions}")
            log_txt("=" * 100)
            log_txt(f"Q: {question}")
            log_txt(f"Ref_Target: {reference_answer}")
            log_txt("-" * 80)
            
            print(f"\n[Progress] Processing Question {q_idx+1}/{total_questions}: {question[:60]}...")

            # Test each config
            for config_name, weights in zip(config_names, alpha_configs):
                log_txt("")
                log_txt(f"Config: {config_name}")
                log_txt(f"Weights: {weights}")
                
                try:
                    response, similarity = pick_best_candidate_with_config(
                        system_prompt, question, model_pipelines, weights, embedding_model
                    )
                    
                    if response and len(response) > 10:
                        log_txt(f"Response: {response}")
                        log_txt(f"Similarity Score: {similarity:.4f}")
                        print(f"  ✓ {config_name} complete (Sim: {similarity:.4f})")
                    else:
                        log_txt(f"Response: [ERROR: Invalid or empty response generated]")
                        log_txt(f"Similarity Score: 0.0000")
                        print(f"  ⚠ {config_name} generated invalid response")
                    
                except Exception as e:
                    log_txt(f"ERROR in {config_name}: {e}")
                    log_txt(f"Similarity Score: 0.0000")
                    print(f"  ✗ {config_name} failed: {e}")
            
            log_txt("")
            
            # --- UPLOAD CHECK ---
            if (q_idx + 1) % 5 == 0 or (q_idx + 1) == total_questions:
                upload_message = f"Checkpoint: Results after Q{q_idx + 1}/{total_questions}"
                upload_to_github(TXT_FILENAME, upload_message, GITHUB_TOKEN) 

        # --- Footer ---
        log_txt("=" * 100)
        log_txt("COMPARISON COMPLETE")
        log_txt("=" * 100)
        log_txt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_txt(f"Total questions processed: {total_questions}")
        if failed_models:
            log_txt(f"Failed models: {failed_models}")

        print(f"\n{'='*50}")
        print("ALL RESULTS SAVED AND UPLOADED!")
        print(f"{'='*50}")
        print(f"Results file: {TXT_FILENAME}")

    except Exception as e:
        log_txt(f"CRITICAL ERROR: {str(e)}")
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# --- Run ---
if __name__ == "__main__":
    print("="*50)
    print("Starting Multi-Configuration Ensemble Comparison")
    print("Mode: 4-BIT QUANTIZATION")
    print(f"GPU Batch Size: {BATCH_SIZE}")
    print(f"Results will be saved to: {TXT_FILENAME}")
    print("="*50)

    if not HF_TOKEN or not GITHUB_TOKEN:
        print("\n!!! WARNING !!!")
        print("HF_TOKEN or GITHUB_TOKEN is empty. Please fill them in for full functionality.")
        if not GITHUB_TOKEN:
             print("GitHub uploads will be skipped.")
        print("!!! WARNING !!!\n")

    run_multi_config_comparison()
