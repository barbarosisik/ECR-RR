#!/usr/bin/env python3
"""
Simple evaluation script for the RL-enhanced Llama2 model.
"""

import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL-enhanced Llama2 model")
    parser.add_argument("--base_model", type=str, default="/data1/s3905993/ECRHMAS/src/models/llama2_chat",
                       help="Base Llama2 model path")
    parser.add_argument("--rl_model", type=str, default="models/rl_enhanced_llama2_optimized/checkpoint-5000",
                       help="RL-enhanced model checkpoint path")
    parser.add_argument("--test_file", type=str, default="src_emo/data/redial/test_data_processed.jsonl",
                       help="Test data file")
    parser.add_argument("--output_file", type=str, default="results/rl_enhanced_evaluation.json",
                       help="Output file for results")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to evaluate")
    parser.add_argument("--max_gen_len", type=int, default=150,
                       help="Maximum generation length")
    return parser.parse_args()

def load_rl_model(base_model_path, rl_model_path):
    """Load the RL-enhanced model."""
    print(f"Loading base model: {base_model_path}")
    print(f"Loading RL-enhanced model from: {rl_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, rl_model_path)
    model.eval()
    
    return model, tokenizer

def format_conversation_for_llama(context, response=None):
    """Format conversation for Llama2 chat model."""
    messages = []
    for i, turn in enumerate(context):
        if i % 2 == 0:
            messages.append({"role": "user", "content": turn})
        else:
            messages.append({"role": "assistant", "content": turn})
    
    if response:
        messages.append({"role": "assistant", "content": response})
    
    formatted = ""
    for message in messages:
        if message["role"] == "user":
            formatted += f"[INST] {message['content']} [/INST]"
        else:
            formatted += f" {message['content']}"
    
    return formatted

def compute_bleu_score(predicted, reference, smoothing=True):
    """Compute BLEU score with smoothing."""
    if smoothing:
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu([reference.split()], predicted.split(), smoothing_function=smoothing_function)
    else:
        return sentence_bleu([reference.split()], predicted.split())

def compute_distinct_score(text, n=1):
    """Compute Distinct-N score."""
    if not text.strip():
        return 0.0
    
    words = text.split()
    if len(words) < n:
        return 0.0
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    
    if not ngrams:
        return 0.0
    
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)

def load_test_data(file_path, num_samples=50):
    """Load test data."""
    print(f"Loading test data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")
    return data

def evaluate_model(model, tokenizer, test_data, max_gen_len=150):
    """Evaluate the model on test data."""
    print("Starting evaluation...")
    
    results = []
    bleu_scores = []
    distinct_1_scores = []
    distinct_2_scores = []
    response_lengths = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        context = sample['context']
        reference = sample['resp']
        
        # Format conversation
        formatted_input = format_conversation_for_llama(context)
        
        # Tokenize input
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + max_gen_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated response
        response_ids = generated_ids[0][inputs['input_ids'].shape[1]:]
        generated_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Calculate metrics
        bleu = compute_bleu_score(generated_response, reference)
        distinct_1 = compute_distinct_score(generated_response, 1)
        distinct_2 = compute_distinct_score(generated_response, 2)
        
        # Store results
        result = {
            'sample_id': i,
            'context': context,
            'reference': reference,
            'generated': generated_response,
            'bleu': bleu,
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'response_length': len(generated_response.split())
        }
        
        results.append(result)
        bleu_scores.append(bleu)
        distinct_1_scores.append(distinct_1)
        distinct_2_scores.append(distinct_2)
        response_lengths.append(len(generated_response.split()))
    
    # Calculate summary statistics
    summary = {
        'num_samples': len(test_data),
        'bleu_1_mean': sum(bleu_scores) / len(bleu_scores),
        'bleu_1_std': torch.std(torch.tensor(bleu_scores, dtype=torch.float32)).item(),
        'distinct_1_mean': sum(distinct_1_scores) / len(distinct_1_scores),
        'distinct_1_std': torch.std(torch.tensor(distinct_1_scores, dtype=torch.float32)).item(),
        'distinct_2_mean': sum(distinct_2_scores) / len(distinct_2_scores),
        'distinct_2_std': torch.std(torch.tensor(distinct_2_scores, dtype=torch.float32)).item(),
        'avg_response_length': sum(response_lengths) / len(response_lengths),
        'response_length_std': torch.std(torch.tensor(response_lengths, dtype=torch.float32)).item()
    }
    
    return results, summary

def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_rl_model(args.base_model, args.rl_model)
    
    # Load test data
    test_data = load_test_data(args.test_file, args.num_samples)
    
    # Evaluate model
    results, summary = evaluate_model(model, tokenizer, test_data, args.max_gen_len)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_data = {
        'summary': summary,
        'results': results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {summary['num_samples']}")
    print(f"BLEU-1: {summary['bleu_1_mean']:.4f} ± {summary['bleu_1_std']:.4f}")
    print(f"Distinct-1: {summary['distinct_1_mean']:.4f} ± {summary['distinct_1_std']:.4f}")
    print(f"Distinct-2: {summary['distinct_2_mean']:.4f} ± {summary['distinct_2_std']:.4f}")
    print(f"Average response length: {summary['avg_response_length']:.1f} ± {summary['response_length_std']:.1f} words")
    print("="*60)
    print(f"Results saved to {args.output_file}")
    
    # Show sample outputs
    print("\nSAMPLE OUTPUTS:")
    print("-" * 50)
    for i, result in enumerate(results[:3]):
        print(f"Sample {i+1}:")
        print(f"Context: {' '.join(result['context'][:2])}...")
        print(f"Reference: {result['reference']}")
        print(f"Generated: {result['generated']}")
        print(f"BLEU: {result['bleu']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 