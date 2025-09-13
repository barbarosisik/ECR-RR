#!/usr/bin/env python3
"""
ECR Results Evaluation Script
Evaluates the generated ECR results using the existing evaluators
"""

import json
import torch
from evaluate_conv import ConvEvaluator
from transformers import AutoTokenizer
import argparse

def load_ecr_results(file_path):
    """Load ECR inference results"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def evaluate_ecr_conversation(results_file, tokenizer_name="microsoft/DialoGPT-small"):
    """Evaluate conversation quality of ECR results"""
    print("=== ECR Conversation Evaluation ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load results
    results = load_ecr_results(results_file)
    print(f"Loaded {len(results)} results from {results_file}")
    
    # Prepare data for evaluation
    preds = []
    labels = []
    
    for result in results:
        # Tokenize predictions and labels
        pred_text = result.get('pred', '')
        label_text = result.get('label', '')
        
        if pred_text and label_text:
            pred_tokens = tokenizer.encode(pred_text, return_tensors='pt', truncation=True, max_length=150)
            label_tokens = tokenizer.encode(label_text, return_tensors='pt', truncation=True, max_length=150)
            
            preds.append(pred_tokens)
            labels.append(label_tokens)
    
    if not preds:
        print("No valid predictions found!")
        return
    
    # Pad sequences
    max_len = max(max(p.shape[1] for p in preds), max(l.shape[1] for l in labels))
    
    padded_preds = []
    padded_labels = []
    
    for pred, label in zip(preds, labels):
        # Pad prediction
        if pred.shape[1] < max_len:
            padding = torch.full((1, max_len - pred.shape[1]), tokenizer.pad_token_id, dtype=pred.dtype)
            pred = torch.cat([pred, padding], dim=1)
        padded_preds.append(pred)
        
        # Pad label
        if label.shape[1] < max_len:
            padding = torch.full((1, max_len - label.shape[1]), tokenizer.pad_token_id, dtype=label.dtype)
            label = torch.cat([label, padding], dim=1)
        padded_labels.append(label)
    
    # Stack tensors
    preds_tensor = torch.cat(padded_preds, dim=0)
    labels_tensor = torch.cat(padded_labels, dim=0)
    
    # Evaluate
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=None)
    evaluator.evaluate(preds_tensor, labels_tensor, log=False)
    metrics = evaluator.report()
    
    print("\n=== ECR Conversation Evaluation Results ===")
    print(f"BLEU-1: {metrics['bleu@1']:.4f}")
    print(f"BLEU-2: {metrics['bleu@2']:.4f}")
    print(f"BLEU-3: {metrics['bleu@3']:.4f}")
    print(f"BLEU-4: {metrics['bleu@4']:.4f}")
    print(f"DIST-1: {metrics['dist@1']:.2f}")
    print(f"DIST-2: {metrics['dist@2']:.2f}")
    print(f"DIST-3: {metrics['dist@3']:.2f}")
    print(f"DIST-4: {metrics['dist@4']:.2f}")
    print(f"Item Ratio: {metrics['item_ratio']:.4f}")
    print(f"Total Sentences: {metrics['sent_cnt']}")
    
    return metrics

def analyze_ecr_quality(results_file):
    """Analyze the quality of ECR responses"""
    print("\n=== ECR Response Quality Analysis ===")
    
    results = load_ecr_results(results_file)
    
    # Analyze response characteristics
    total_responses = len(results)
    empty_responses = sum(1 for r in results if not r.get('pred', '').strip())
    short_responses = sum(1 for r in results if len(r.get('pred', '').split()) < 5)
    long_responses = sum(1 for r in results if len(r.get('pred', '').split()) > 50)
    
    print(f"Total responses: {total_responses}")
    print(f"Empty responses: {empty_responses} ({empty_responses/total_responses*100:.2f}%)")
    print(f"Short responses (<5 words): {short_responses} ({short_responses/total_responses*100:.2f}%)")
    print(f"Long responses (>50 words): {long_responses} ({long_responses/total_responses*100:.2f}%)")
    
    # Show some examples
    print("\n=== Sample ECR Responses ===")
    for i, result in enumerate(results[:5]):
        print(f"\nSample {i+1}:")
        print(f"Input: {result.get('input', 'N/A')[:100]}...")
        print(f"Prediction: {result.get('pred', 'N/A')}")
        print(f"Label: {result.get('label', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ECR results")
    parser.add_argument("--results_file", default="save/redial_gen/emp_test.jsonl", 
                       help="Path to ECR results file")
    parser.add_argument("--tokenizer", default="microsoft/DialoGPT-small",
                       help="Tokenizer to use for evaluation")
    
    args = parser.parse_args()
    
    # Evaluate conversation quality
    conv_metrics = evaluate_ecr_conversation(args.results_file, args.tokenizer)
    
    # Analyze response quality
    analyze_ecr_quality(args.results_file)
    
    print("\n=== ECR Evaluation Complete ===")

if __name__ == "__main__":
    main() 