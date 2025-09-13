#!/usr/bin/env python3
"""
Test Llama2 scoring with explicit reasoning for each score (5 samples)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import os
from typing import Dict, List, Optional
import numpy as np
import time
from datetime import datetime, timedelta

# Paths
REAL_DATA_PATH = "../ECRHMAS/data/redial_gen/train_data_processed.jsonl"
MODEL_NAME = "/data1/s3905993/ECRHMAS/src/models/llama2_chat"
OUTPUT_FILE = "llama2_scored_full_dataset.jsonl"

def create_llama2_prompt(context: List[str], response: str) -> str:
    """Create a Llama2 prompt for quality scoring (FAST, NO REASONING)"""
    context_str = " ".join([turn.strip() for turn in context if turn.strip()])
    prompt = (
        "Please evaluate the quality of the following conversational response. "
        "Rate each aspect on a scale of 0.0 to 1.0.\n\n"
        f"Context: {context_str}\n"
        f"Response: {response}\n\n"
        "Please provide scores for the following aspects:\n"
        "- Empathy: How well does the response show understanding and emotional support?\n"
        "- Informativeness: How much useful information does the response provide?\n"
        "- Recommendation quality: If a movie or item is recommended in the response (taking the context into account), rate the quality of the recommendation (0.0-1.0). If no recommendation is made, set this score to 0.0.\n"
        "- Engagement: How engaging and interesting is the response?\n"
        "- Overall quality: Overall assessment of the response quality\n\n"
        "Respond with ONLY a JSON object with these keys: empathy_score, informativeness_score, recommendation_score, engagement_score, overall_score. No explanations, no extra text."
    )
    return prompt

def get_llama2_scores(prompt: str, model, tokenizer) -> Optional[Dict]:
    """Get quality scores from Llama2-Chat using chat template (FAST, NO REASONING)"""
    try:
        system_message = (
            "You are a quality assessment expert. Respond with ONLY a valid JSON object. "
            "No explanations, no extra text."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for consistent outputs
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            return None
        json_str = result[json_start:json_end]
        scores = json.loads(json_str)
        # Validate scores (set to 0.5 if invalid)
        for key in [
            'empathy_score', 'informativeness_score', 'recommendation_score', 'engagement_score', 'overall_score']:
            value = scores.get(key, 0.5)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                scores[key] = 0.5
        return scores
    except Exception as e:
        print(f"Error getting scores: {e}")
        return None

# Add BLEU computation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.strip().split()
    hypothesis_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    try:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    except Exception:
        bleu = 0.0
    return float(bleu)

def process_dataset(input_file: str, output_file: str, max_samples: Optional[int] = None, 
                   start_from: int = 0, model=None, tokenizer=None):
    """Process the full dataset and generate scores (with ETA reporting)"""
    print(f"Processing dataset: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print(f"Starting from: {start_from}")
    processed_count = 0
    success_count = 0
    start_time = time.time()
    total_samples = None
    # Try to estimate total samples
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)
        if max_samples:
            total_samples = min(total_samples - start_from, max_samples)
        else:
            total_samples = total_samples - start_from
    except Exception:
        pass
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'a', encoding='utf-8') as fout:
        # Skip to start_from
        for _ in range(start_from):
            next(fin)
        for line in tqdm(fin, desc="Processing conversations"):
            if max_samples and processed_count >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                context = item.get('context', [])
                response = item.get('resp', '')
                target = item.get('target', '') or item.get('gt_resp', '') or item.get('reference', '')
                if not response.strip():
                    continue
                prompt = create_llama2_prompt(context, response)
                scores = get_llama2_scores(prompt, model, tokenizer)
                if scores:
                    # Compute BLEU using real metric if reference is available
                    if target:
                        bleu = compute_bleu(target, response)
                    else:
                        bleu = None
                    scores['bleu_score'] = bleu if bleu is not None else 0.0
                    item['quality_scores'] = scores
                    item['llama2_prompt'] = prompt
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    success_count += 1
                processed_count += 1
                # ETA reporting every 10 samples
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    if total_samples and rate > 0:
                        eta_sec = (total_samples - processed_count) / rate
                        finish_time = datetime.now() + timedelta(seconds=eta_sec)
                        print(f"[PROGRESS] {processed_count}/{total_samples} processed | Elapsed: {elapsed/60:.1f} min | ETA: {eta_sec/60:.1f} min | Finish: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        print(f"[PROGRESS] {processed_count} processed | Elapsed: {elapsed/60:.1f} min | Rate: {rate:.2f} samples/sec")
            except Exception as e:
                print(f"Error processing line {processed_count}: {e}")
                continue
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Successful: {success_count}")
    print(f"Success rate: {success_count/processed_count*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Test Llama2 scoring with reasoning (5 samples)')
    parser.add_argument('--max', type=int, default=5, help='Maximum number of samples to process (default: 5)')
    parser.add_argument('--start', type=int, default=0, help='Start from this sample number (for resuming)')
    parser.add_argument('--input', type=str, default=REAL_DATA_PATH, help='Input data file path')
    parser.add_argument('--output', type=str, default="llama2_scored_reasoning_test.jsonl", help='Output file path')
    args = parser.parse_args()
    print("=== LLAMA2 QUALITY SCORING WITH REASONING (TEST) ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max samples: {args.max}")
    print(f"Start from: {args.start}")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    print("\nLoading Llama2 model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    process_dataset(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max,
        start_from=args.start,
        model=model,
        tokenizer=tokenizer
    )
if __name__ == "__main__":
    main() 