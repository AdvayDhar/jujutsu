# --- Multi-Configuration Ensemble Comparison (4-bit Quantization) ---
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import torch
import torch.nn.functional as F
import random
import numpy as np
import re
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

# --- YOUR QUESTIONS ---
YOUR_QUESTIONS = [
    "What is a panic attack and how can I deal with it?",
    "How can I manage anxiety in stressful situations?",
    "I feel depressed, what should I do?",
    "What are some coping strategies for stress?",
    "How do I know if I need therapy?",
    "What are the signs of burnout?",
    "How can I improve my sleep when I'm anxious?",
    "What should I do when I feel overwhelmed?",
    "How can I support a friend with depression?",
    "What is the difference between sadness and depression?",
    "Is anxiety in a black man different from a white man's?",
    "Is feeling anxious around white cops normal? I am black",
    "I am white. is feeling cautious of black men normal?"
]

# --- Configuration ---
# Your HuggingFace token
HF_TOKEN = ""
if HF_TOKEN:
    login(token=HF_TOKEN)
    
# --- Alpha Configurations ---
alpha_configs = [
    [0.100, 0.100, 0.100, 0.600, 0.100],  # Config 1: Best Perform
    [0.587, 0.113, 0.100, 0.100, 0.100],  # Config 2: best Fairness
    [0.100, 0.600, 0.100, 0.100, 0.100],  # Config 3: best cost
    [0.195, 0.131, 0.177, 0.243, 0.253],  # Config 4: best diversity
    [0.391, 0.309, 0.100, 0.100, 0.100],  # Config 5: Balanced output
]

config_names = [
    "Best Perform",
    "Best Fairness", 
    "Best cost",
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
    "gemma2b": "advy/gemma2b-mental-health-assistant",
    "distilgpt2": "advy/distilgpt2-mental-health-assistant", 
    "phi2": "advy/phi2-mental-health-assistant",
    "mistral": "advy/mistral-mental-health-assistant",
    "tinyllama": "advy/tinyllama-mental-health-assistant"
}

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

# Load models WITH 4-bit quantization
def load_ensemble_models():
    """Load all ensemble models with 4-bit quantization"""
    model_pipelines = {}
    failed_models = []

    for name, mid in models_info.items():
        try:
            print(f"Loading {name} with 4-bit quantization...")
            tok = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=False)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
            # With quantization config
            model_kwargs = {"trust_remote_code": False, "torch_dtype": dtype}
            if device == "cuda" and bnb_config:
                model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
            else:
                model_kwargs["device_map"] = "auto" if device == "cuda" else None
                
            model = AutoModelForCausalLM.from_pretrained(mid, **model_kwargs)
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
# Common instruction prompt for all models
INSTRUCTION_TEMPLATE = """Answer the following question in under 100 words. Provide a complete, helpful response.

Question: {question}

Answer:"""

GEN_KW = dict(
    max_new_tokens=200,  # Enough for ~100 words + buffer
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.05,
    return_full_text=False  # Critical: exclude prompt from output
)

def clean_response(response: str) -> str:
    """Clean response while preserving complete content"""
    if not response or not response.strip():
        return ""
    
    # Remove extra whitespace but preserve sentence structure
    response = re.sub(r'\s+', ' ', response).strip()
    
    # Remove any common prefixes that models might add
    prefixes_to_remove = [
        "Assistant:", "Answer:", "Response:", "A:", 
        "Human:", "Question:", "Q:",
        "Here is", "Here's"
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            # Check again after removal (sometimes there are multiple)
            break
    
    # Remove "Here is the answer:" type patterns at the start
    response = re.sub(r'^(Here is (the|my|an) (answer|response)[\s:]*)', '', response, flags=re.IGNORECASE).strip()
    
    # Ensure response ends with proper punctuation
    if response and response[-1] not in '.!?':
        # Find the last sentence-ending punctuation
        last_period = max(
            response.rfind('.'),
            response.rfind('!'),
            response.rfind('?')
        )
        
        # Only truncate if we found punctuation and it's in the latter half of response
        if last_period > len(response) * 0.3:
            response = response[:last_period + 1]
        else:
            # If no good punctuation found, just add a period
            response = response.rstrip() + '.'
    
    return response

def get_model_output(pipe, prompt: str) -> str:
    """Generate response from a single model with instruction template"""
    try:
        # Format prompt with instruction template
        formatted_prompt = INSTRUCTION_TEMPLATE.format(question=prompt)
        
        # Generate response
        out = pipe(
            formatted_prompt,
            **GEN_KW, 
            pad_token_id=pipe.tokenizer.eos_token_id
        )[0]["generated_text"]
        
        # Clean and validate response
        cleaned = clean_response(out.strip())
        
        # Validate we have a meaningful response
        if not cleaned or len(cleaned) < 10:
            print(f"Warning: Generated response too short or empty")
            return ""
        
        return cleaned
        
    except Exception as e:
        print(f"Error generating output: {e}")
        return ""

def weighted_average_embedding(responses, weights, embedding_model):
    """Compute weighted average of response embeddings"""
    if len(responses) != len(weights):
        raise ValueError("responses and weights must match")
    
    valid = [(r, w) for r, w in zip(responses, weights) if r.strip()]
    if not valid:
        raise ValueError("No valid responses")
    
    responses, weights = zip(*valid)
    embs = embedding_model.encode(list(responses), convert_to_tensor=True, normalize_embeddings=True)
    w = torch.tensor(weights, dtype=embs.dtype, device=embs.device).unsqueeze(1)
    weighted = torch.sum(embs * w, dim=0) / torch.sum(w)
    
    return F.normalize(weighted, p=2, dim=0)

def generate_diverse_candidates(model_pipes, prompt: str, candidates_per_model: int = 4):
    """Generate diverse candidates from all models"""
    all_candidates = []
    
    for model_name, pipe in model_pipes.items():
        for i in range(candidates_per_model):
            try:
                response = get_model_output(pipe, prompt)
                # Validate response quality
                if response and response.strip() and len(response) > 20:
                    all_candidates.append(response)
            except Exception as e:
                print(f"Error generating candidate from {model_name} (attempt {i+1}): {e}")
                continue
    
    # Remove exact duplicates
    seen = set()
    unique_candidates = []
    for cand in all_candidates:
        if cand not in seen:
            seen.add(cand)
            unique_candidates.append(cand)
    
    print(f"Generated {len(unique_candidates)} unique candidates from {len(all_candidates)} total")
    return unique_candidates

def pick_best_candidate_with_config(question, model_pipes, weights, embedding_model, candidates_per_model=4):
    """Pick best candidate using specific weight configuration"""
    # Generate diverse candidates
    candidates = generate_diverse_candidates(model_pipes, question, candidates_per_model)
    
    if not candidates:
        print("Warning: No candidates generated!")
        return "", 0.0
    
    # Generate ensemble responses for comparison
    ensemble_responses, ensemble_weights = [], []
    for (name, pipe), w in zip(model_pipes.items(), weights):
        try:
            resp = get_model_output(pipe, question)
            if resp and resp.strip():
                ensemble_responses.append(resp)
                ensemble_weights.append(w)
        except Exception as e:
            print(f"Error getting response from {name}: {e}")
            continue
    
    if not ensemble_responses:
        print("Warning: No ensemble responses generated, returning first candidate")
        return candidates[0], 0.0
    
    # Find best candidate using ensemble comparison
    ensemble_vec = weighted_average_embedding(ensemble_responses, ensemble_weights, embedding_model)
    cand_embs = embedding_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    sims = F.cosine_similarity(cand_embs, ensemble_vec.unsqueeze(0))
    best_idx = torch.argmax(sims).item()
    
    return candidates[best_idx], sims[best_idx].item()

# --- Main Comparison Function ---
def run_multi_config_comparison():
    """Run comparison across multiple ensemble configurations"""
    
    txt_filename = "ensembleresults.txt"
    
    # --- Logging helper ---
    def log_txt(line: str = ""):
        with open(txt_filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)
    
    # Clear previous results
    if os.path.exists(txt_filename):
        os.remove(txt_filename)
    
    # --- Header ---
    log_txt("=" * 100)
    log_txt("MULTI-CONFIGURATION ENSEMBLE COMPARISON (4-BIT QUANTIZATION)")
    log_txt("=" * 100)
    log_txt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_txt(f"Device: {device}")
    log_txt(f"Data type: {dtype}")
    log_txt(f"Quantization: 4-bit NF4")
    log_txt(f"Total Questions: {len(YOUR_QUESTIONS)}")
    log_txt(f"Instruction: All models instructed to answer in under 100 words")
    log_txt("")
    log_txt("Alpha Configurations:")
    for name, config in zip(config_names, alpha_configs):
        log_txt(f"  {name}: {config}")
    log_txt("")

    try:
        # Load models
        print("Loading models with 4-bit quantization...")
        model_pipelines, failed_models = load_ensemble_models()
        
        # Adjust alpha configs if needed
        if len(alpha_configs[0]) != len(model_pipelines):
            num_models = len(model_pipelines)
            alpha_configs[:] = [[1.0 / num_models] * num_models for _ in alpha_configs]
            log_txt(f"Adjusted alpha configs to match {num_models} loaded models")
            log_txt("")

        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"\nProcessing {len(YOUR_QUESTIONS)} questions...")
        
        for q_idx, question in enumerate(YOUR_QUESTIONS):
            log_txt("=" * 100)
            log_txt(f"QUESTION {q_idx+1}/{len(YOUR_QUESTIONS)}")
            log_txt("=" * 100)
            log_txt(f"Q: {question}")
            log_txt("-" * 80)
            
            print(f"\n[Progress] Processing Question {q_idx+1}/{len(YOUR_QUESTIONS)}: {question[:60]}...")

            # Test each config
            for config_name, weights in zip(config_names, alpha_configs):
                log_txt("")
                log_txt(f"Config: {config_name}")
                log_txt(f"Weights: {weights}")
                
                try:
                    response, similarity = pick_best_candidate_with_config(
                        question, model_pipelines, weights, embedding_model, candidates_per_model=4
                    )
                    
                    # Validate response before logging
                    if response and len(response) > 10:
                        log_txt(f"Response: {response}")
                        log_txt(f"Similarity Score: {similarity:.4f}")
                        print(f"  ✓ {config_name} complete (response length: {len(response)} chars)")
                    else:
                        log_txt(f"Response: [ERROR: Invalid or empty response generated]")
                        log_txt(f"Similarity Score: 0.0000")
                        print(f"  ⚠ {config_name} generated invalid response")
                    
                except Exception as e:
                    log_txt(f"ERROR in {config_name}: {e}")
                    print(f"  ✗ {config_name} failed: {e}")
            
            log_txt("")
            print(f"[✓] Question {q_idx+1}/{len(YOUR_QUESTIONS)} results saved to {txt_filename}")

        log_txt("")
        log_txt("=" * 100)
        log_txt("COMPARISON COMPLETE")
        log_txt("=" * 100)
        log_txt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_txt(f"Total questions processed: {len(YOUR_QUESTIONS)}")
        if failed_models:
            log_txt(f"Failed models: {failed_models}")

        print(f"\n{'='*50}")
        print("ALL RESULTS SAVED!")
        print(f"{'='*50}")
        print(f"Results file: {txt_filename}")

    except Exception as e:
        log_txt(f"CRITICAL ERROR: {str(e)}")
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    return txt_filename


# --- Run ---
if __name__ == "__main__":
    print("="*50)
    print("Starting Multi-Configuration Ensemble Comparison")
    print("Mode: 4-BIT QUANTIZATION")
    print("="*50)
    print(f"Total questions to process: {len(YOUR_QUESTIONS)}")
    print(f"Device: {device}")
    print(f"Precision: {dtype}")
    print(f"Instruction: All models answer in under 100 words")
    print("="*50)
    
    result_file = run_multi_config_comparison()
    
    print(f"\n{'='*50}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*50}")
    print(f"All results saved to: {result_file}")
