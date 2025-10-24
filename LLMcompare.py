# --- Multi-Configuration Ensemble vs Gemini Comparison with File Output ---
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from huggingface_hub import login
import torch
import torch.nn.functional as F
import random
import numpy as np
import re
import warnings
import google.generativeai as genai
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
import time
import nltk
import sys
from datetime import datetime


warnings.filterwarnings("ignore")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- File Output Class ---
class FileLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, text: str):
        """Write to both console and file"""
        print(text)
        self.file.write(text + '\n')
        self.file.flush()
        
    def close(self):
        self.file.close()

# --- Configuration ---
# Your HuggingFace token
HF_TOKEN = ""
if HF_TOKEN:
    login(token=HF_TOKEN)

# Add your Gemini API key here
GEMINI_API_KEY = "AIzaSyDArrMPotArPmWtz4HWngWY-p_2DKLwUOo"  # Replace with your actual key
if GEMINI_API_KEY.strip():  # only checks if it's not empty
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    print("Warning: Gemini API key not configured!")
    
    
    
# --- Alpha Configurations ---
# Define different weight configurations for your ensemble
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

# --- Dataset Processing ---
def parse_conversation_with_answers(text):
    """Extract questions and reference answers from dataset"""
    patterns = [
        r'<HUMAN>:\s*(.*?)\s*<ASSISTANT>:\s*(.*?)(?=<HUMAN>|$)',
        r'Human:\s*(.*?)\s*Assistant:\s*(.*?)(?=Human:|$)',
        r'User:\s*(.*?)\s*Bot:\s*(.*?)(?=User:|$)',
    ]
    
    qa_pairs = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            for question, answer in matches:
                question = re.sub(r'\s+', ' ', question.strip())
                answer = re.sub(r'\s+', ' ', answer.strip())
                
                if (10 < len(question) < 500 and 10 < len(answer) < 1000):
                    greeting_phrases = ['hello', 'hi there', 'good morning']
                    if not any(phrase in question.lower() for phrase in greeting_phrases):
                        qa_pairs.append({
                            'question': question,
                            'reference_answer': answer
                        })
            break
    return qa_pairs

def load_mental_health_qa_dataset(num_samples=20):
    """Load QA pairs from mental health dataset"""
    try:
        dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")
        print(f"Loaded dataset with {len(dataset)} examples")
        
        all_qa_pairs = []
        for i, example in enumerate(dataset.shuffle(seed=42).select(range(min(num_samples * 3, len(dataset))))):
            qa_pairs = parse_conversation_with_answers(example["text"])
            all_qa_pairs.extend(qa_pairs)
            if len(all_qa_pairs) >= num_samples:
                break
        
        seen_questions = set()
        unique_pairs = []
        for pair in all_qa_pairs:
            if pair['question'] not in seen_questions:
                seen_questions.add(pair['question'])
                unique_pairs.append(pair)
        
        result = unique_pairs[:num_samples]
        print(f"Returning {len(result)} QA pairs")
        return result
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback dataset
        return [
            {
                'question': "What is a panic attack and how can I manage one?",
                'reference_answer': "A panic attack is a sudden episode of intense fear that triggers severe physical reactions. To manage one, try deep breathing, grounding techniques like the 5-4-3-2-1 method, and remind yourself that it will pass."
            },
            {
                'question': "How can I cope with anxiety on a daily basis?",
                'reference_answer': "Daily anxiety can be managed through regular exercise, mindfulness meditation, maintaining a routine, limiting caffeine, and practicing relaxation techniques. Consider therapy if anxiety interferes with daily activities."
            },
            {
                'question': "I feel depressed lately, what should I do?",
                'reference_answer': "If you're feeling depressed, it's important to reach out for support. Consider talking to a therapist, maintaining social connections, engaging in regular physical activity, and establishing healthy sleep patterns. Professional help is available."
            },
            {
                'question': "What are some effective coping strategies for stress?",
                'reference_answer': "Effective stress coping strategies include deep breathing exercises, regular physical activity, time management, setting boundaries, practicing mindfulness, maintaining social connections, and engaging in hobbies you enjoy."
            },
            {
                'question': "How do I know if I need professional therapy?",
                'reference_answer': "Consider therapy if you're experiencing persistent sadness, anxiety affecting daily life, relationship problems, trauma, substance use issues, or if you're feeling overwhelmed and unable to cope with daily stressors."
            }
        ]

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

# Quantization config
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )
else:
    bnb_config = None

# Load models
def load_ensemble_models():
    """Load all ensemble models"""
    model_pipelines = {}
    failed_models = []

    for name, mid in models_info.items():
        try:
            tok = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=False)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
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
GEN_KW = dict(
    max_new_tokens=150,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.05,
    return_full_text=False
)

def clean_response(response: str) -> str:
    response = re.sub(r'\s+', ' ', response).strip()
    sentences = response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        response = '.'.join(sentences[:-1]) + '.'
    return response

def get_model_output(pipe, prompt: str) -> str:
    try:
        formatted_prompt = f"Human: {prompt}\nAssistant: "
        out = pipe(formatted_prompt, **GEN_KW, pad_token_id=pipe.tokenizer.eos_token_id)[0]["generated_text"]
        return clean_response(out[len(formatted_prompt):].strip())
    except Exception as e:
        return ""

def weighted_average_embedding(responses, weights, embedding_model):
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
                if response.strip() and len(response) > 10:
                    all_candidates.append(response)
            except Exception as e:
                continue
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for cand in all_candidates:
        if cand not in seen:
            seen.add(cand)
            unique_candidates.append(cand)
    
    return unique_candidates

def pick_best_candidate_with_config(question, model_pipes, weights, embedding_model, candidates_per_model=4):
    """Pick best candidate using specific weight configuration"""
    # Generate diverse candidates
    candidates = generate_diverse_candidates(model_pipes, question, candidates_per_model)
    
    if not candidates:
        return "", 0.0
    
    # Generate ensemble responses for comparison
    ensemble_responses, ensemble_weights = [], []
    for (name, pipe), w in zip(model_pipes.items(), weights):
        try:
            resp = get_model_output(pipe, question)
            if resp.strip():
                ensemble_responses.append(resp)
                ensemble_weights.append(w)
        except Exception as e:
            continue
    
    if not ensemble_responses:
        return candidates[0], 0.0
    
    # Find best candidate using ensemble comparison
    ensemble_vec = weighted_average_embedding(ensemble_responses, ensemble_weights, embedding_model)
    cand_embs = embedding_model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    sims = F.cosine_similarity(cand_embs, ensemble_vec.unsqueeze(0))
    best_idx = torch.argmax(sims).item()
    
    return candidates[best_idx], sims[best_idx].item()

# --- Gemini API ---
def get_gemini_response(prompt: str, max_retries: int = 3) -> str:
    """Get response from Gemini API"""
    if not gemini_model:
        return "Gemini API not configured"
        
    formatted_prompt = f"""You are a mental health assistant. Please provide a helpful, empathetic, and informative response to the following question:

Question: {prompt}

Please provide a concise but comprehensive answer:"""

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(formatted_prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Gemini API failed after {max_retries} attempts: {str(e)}"
    return "Gemini API failed to generate response"

# --- Evaluation Metrics ---
class MetricsCalculator:
    def __init__(self):
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()
    
    def calculate_rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        try:
            scores = self.rouge.get_scores(generated, reference)[0]
            return {
                'rouge1_f': scores['rouge-1']['f'],
                'rouge2_f': scores['rouge-2']['f'],
                'rougeL_f': scores['rouge-l']['f']
            }
        except:
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
    
    def calculate_bleu_score(self, reference: str, generated: str) -> float:
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            gen_tokens = nltk.word_tokenize(generated.lower())
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing.method1)
            return bleu_score
        except:
            return 0.0
    
    def calculate_all_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        if not reference.strip() or not generated.strip():
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0, 'bleu': 0.0}
        
        rouge_scores = self.calculate_rouge_scores(reference, generated)
        bleu_score = self.calculate_bleu_score(reference, generated)
        
        return {**rouge_scores, 'bleu': bleu_score}

import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- Main Comparison Function ---
def run_multi_config_comparison(num_questions: int = 20):
    """Run comparison across multiple ensemble configurations + Gemini 
    with single persistent JSONL and TXT logs (append mode)."""

    # Fixed filenames (no timestamp)
    jsonl_filename = "ensemble_comparison.jsonl"
    txt_filename = "ensemble_comparison.txt"

    # --- Logging helpers ---
    def log_json(entry: dict):
        with open(jsonl_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_txt(line: str = ""):
        with open(txt_filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # --- Header (each run still records its own start timestamp) ---
    header_info = {
        "type": "header",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dtype": str(dtype),
        "alpha_configurations": [
            {"name": name, "weights": config}
            for name, config in zip(config_names, alpha_configs)
        ],
    }
    log_json(header_info)

    log_txt("=" * 100)
    log_txt("MULTI-CONFIGURATION ENSEMBLE vs GEMINI COMPARISON (NEW RUN)")
    log_txt("=" * 100)
    log_txt(f"Timestamp: {header_info['timestamp']}")
    log_txt(f"Device: {header_info['device']}")
    log_txt(f"Data type: {header_info['dtype']}")
    log_txt("")
    log_txt("Alpha Configurations:")
    for cfg in header_info["alpha_configurations"]:
        log_txt(f"  {cfg['name']}: {cfg['weights']}")
    log_txt("")

    try:
        # Load models
        model_pipelines, failed_models = load_ensemble_models()
        if len(alpha_configs[0]) != len(model_pipelines):
            num_models = len(model_pipelines)
            alpha_configs[:] = [[1.0 / num_models] * num_models for _ in alpha_configs]

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        qa_pairs = load_mental_health_qa_dataset(num_questions)
        if not qa_pairs:
            log_json({"type": "error", "message": "No QA pairs loaded!"})
            log_txt("ERROR: No QA pairs loaded!")
            return

        metrics_calc = MetricsCalculator()
        all_results = []

        for q_idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            reference_answer = qa_pair["reference_answer"]

            log_txt("=" * 100)
            log_txt(f"QUESTION {q_idx+1}/{len(qa_pairs)}")
            log_txt("=" * 100)
            log_txt(f"Q: {question}")
            log_txt("")
            log_txt(f"Reference Answer: {reference_answer}")
            log_txt("-" * 80)

            question_results = {
                "type": "question",
                "id": q_idx + 1,
                "question": question,
                "reference_answer": reference_answer,
                "results": {},
            }

            # Ensemble configs
            for config_name, weights in zip(config_names, alpha_configs):
                log_txt("")
                log_txt(f"Config: {config_name}")
                log_txt(f"Weights: {weights}")
                try:
                    response, similarity = pick_best_candidate_with_config(
                        question, model_pipelines, weights, embedding_model, candidates_per_model=4
                    )
                    metrics = metrics_calc.calculate_all_metrics(reference_answer, response)
                    log_txt(f"Response: {response}")
                    log_txt(
                        f"ROUGE-1: {metrics['rouge1_f']:.3f} | "
                        f"ROUGE-2: {metrics['rouge2_f']:.3f} | "
                        f"ROUGE-L: {metrics['rougeL_f']:.3f} | "
                        f"BLEU: {metrics['bleu']:.3f} | "
                        f"Sim: {similarity:.3f}"
                    )
                    question_results["results"][config_name] = {
                        "response": response,
                        "similarity": similarity,
                        **metrics,
                    }
                except Exception as e:
                    log_txt(f"ERROR in {config_name}: {e}")
                    question_results["results"][config_name] = {
                        "response": f"Error: {str(e)}",
                        "similarity": 0.0,
                        "rouge1_f": 0.0,
                        "rouge2_f": 0.0,
                        "rougeL_f": 0.0,
                        "bleu": 0.0,
                    }

            # Gemini
            gemini_response = get_gemini_response(question)
            gemini_metrics = metrics_calc.calculate_all_metrics(reference_answer, gemini_response)
            log_txt("")
            log_txt("Gemini API:")
            log_txt(f"Response: {gemini_response}")
            log_txt(
                f"ROUGE-1: {gemini_metrics['rouge1_f']:.3f} | "
                f"ROUGE-2: {gemini_metrics['rouge2_f']:.3f} | "
                f"ROUGE-L: {gemini_metrics['rougeL_f']:.3f} | "
                f"BLEU: {gemini_metrics['bleu']:.3f}"
            )

            question_results["results"]["Gemini"] = {
                "response": gemini_response,
                **gemini_metrics,
            }

            log_json(question_results)
            all_results.append(question_results)

        # --- FINAL SUMMARY ---
        df_results = pd.DataFrame([
            {"id": q["id"], "config": cfg, **res}
            for q in all_results
            for cfg, res in q["results"].items()
        ])
        summary_data = {}
        for config_name in df_results["config"].unique():
            subset = df_results[df_results["config"] == config_name]
            summary_data[config_name] = {
                "rouge1": subset["rouge1_f"].mean(),
                "rouge2": subset["rouge2_f"].mean(),
                "rougeL": subset["rougeL_f"].mean(),
                "bleu": subset["bleu"].mean(),
                "similarity": subset["similarity"].mean()
                if "similarity" in subset
                else None,
            }

        log_json({
            "type": "summary",
            "summary_data": summary_data,
            "total_questions": len(qa_pairs),
            "failed_models": failed_models,
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        log_txt("")
        log_txt("=" * 100)
        log_txt("FINAL SUMMARY")
        log_txt("=" * 100)
        for name, scores in summary_data.items():
            log_txt(f"{name}:")
            for k, v in scores.items():
                log_txt(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

        print(f"\nStructured logs: {jsonl_filename}")
        print(f"Readable logs:   {txt_filename}")

    except Exception as e:
        log_json({"type": "critical_error", "message": str(e)})
        log_txt(f"CRITICAL ERROR: {str(e)}")

    return jsonl_filename, txt_filename


# --- Run ---
if __name__ == "__main__":
    print("Starting multi-configuration comparison with persistent JSONL + TXT logging...")
    try:
        num_q = int(input("Enter number of questions to process (default 10): ") or "10")
    except:
        num_q = 10

    json_file, txt_file = run_multi_config_comparison(num_questions=num_q)
    print(f"\n{'='*50}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*50}")
    print(f"Readable TXT log appended to: {txt_file}")
    print(f"Structured JSONL log appended to: {json_file}")
