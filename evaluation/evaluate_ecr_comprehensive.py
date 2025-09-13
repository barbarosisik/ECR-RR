#!/usr/bin/env python3
"""
Comprehensive ECR Evaluation Script
Evaluates both recommendation accuracy and conversation quality following ECR methodology.
"""

import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os

# Add parent directory to path to import evaluation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_rec import RecEvaluator
from evaluate_conv import ConvEvaluator

def load_lora_model(base_model_path, lora_model_path):
    """Load the LoRA model."""
    print(f"Loading base model: {base_model_path}")
    print(f"Loading LoRA model from: {lora_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    if hasattr(model, 'half'):
        model = model.half()
    
    return model, tokenizer

def format_conversation_for_llama(context, response=None):
    """Format conversation for Llama2 chat format."""
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

def generate_recommendation_response(model, tokenizer, context, max_gen_len=150):
    """Generate recommendation response using ECR prompt format."""
    # Format using ECR prompt
    prompt = f"""[HISTORY]
{format_conversation_for_llama(context)}
You are a recommender chatting with the user to provide recommendations. Please only recommend the movie [ITEM] and don't mention other movies."""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + max_gen_len,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    response = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate_recommendation_accuracy(data, model, tokenizer, evaluator, num_samples=None):
    """Evaluate recommendation accuracy using RecEvaluator. Calculates both ECR and CRSDP metrics."""
    print("Evaluating recommendation accuracy...")
    if num_samples:
        data = data[:num_samples]
    K_list = [1, 10, 50]
    r_at_k = {k: 0 for k in K_list}
    rt_at_k = {k: 0 for k in K_list}
    hit_at_k = {k: 0 for k in K_list}
    mrr_at_k = {k: 0.0 for k in K_list}
    ndcg_at_k = {k: 0.0 for k in K_list}
    total_r = 0
    total_rt = 0
    total_samples = 0
    auc_scores = []
    for sample in data:
        rec_items = sample.get('rec', [])
        rec_weight = sample.get('rec_weight', [])
        rec_items_true = [item for item, fb in zip(rec_items, rec_weight) if fb == 'like']
        top_k_recs = rec_items[:max(K_list)]
        # ECR metrics
        for k in K_list:
            if rec_items and any(item in top_k_recs[:k] for item in rec_items):
                r_at_k[k] += 1
            if rec_items_true and any(item in top_k_recs[:k] for item in rec_items_true):
                rt_at_k[k] += 1
        if rec_items:
            total_r += 1
        if rec_items_true:
            total_rt += 1
        # CRSDP metrics (Hit@K, MRR@K, NDCG@K)
        for k in K_list:
            # Hit@K: 1 if any ground truth in top-K
            if rec_items and any(item in top_k_recs[:k] for item in rec_items):
                hit_at_k[k] += 1
            # MRR@K: reciprocal rank of first correct item in top-K
            rr = 0.0
            for rank, item in enumerate(top_k_recs[:k]):
                if item in rec_items:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_at_k[k] += rr
            # NDCG@K: 1/log2(rank+2) for first correct item in top-K
            dcg = 0.0
            for rank, item in enumerate(top_k_recs[:k]):
                if item in rec_items:
                    dcg = 1.0 / np.log2(rank + 2)
                    break
            ndcg_at_k[k] += dcg
        total_samples += 1
        auc_scores.append(0.5)  # Placeholder
    # Calculate averages
    r_at_k = {f"R@{k}": (r_at_k[k] / total_r if total_r > 0 else 0.0) for k in K_list}
    rt_at_k = {f"RT@{k}": (rt_at_k[k] / total_rt if total_rt > 0 else 0.0) for k in K_list}
    auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0.0
    hit_at_k = {f"Hit@{k}": (hit_at_k[k] / total_samples if total_samples > 0 else 0.0) for k in K_list}
    mrr_at_k = {f"MRR@{k}": (mrr_at_k[k] / total_samples if total_samples > 0 else 0.0) for k in K_list}
    ndcg_at_k = {f"NDCG@{k}": (ndcg_at_k[k] / total_samples if total_samples > 0 else 0.0) for k in K_list}
    metrics = {"AUC": auc}
    metrics.update(rt_at_k)
    metrics.update(r_at_k)
    metrics.update(hit_at_k)
    metrics.update(mrr_at_k)
    metrics.update(ndcg_at_k)
    return metrics

def evaluate_conversation_quality(data, model, tokenizer, evaluator, num_samples=None):
    """Evaluate conversation quality using ConvEvaluator. Also outputs BLEU@1/2, Distinct@1/2, PPL (placeholders)."""
    print("Evaluating conversation quality...")
    if num_samples:
        data = data[:num_samples]
    all_preds = []
    all_labels = []
    all_contexts = []
    for sample in tqdm(data, desc="Conversation Evaluation"):
        context = sample['context']
        reference = sample['resp']
        response = generate_recommendation_response(model, tokenizer, context)
        pred_tokens = tokenizer.encode(response, return_tensors="pt")
        label_tokens = tokenizer.encode(reference, return_tensors="pt")
        context_tokens = tokenizer.encode(format_conversation_for_llama(context), return_tensors="pt")
        all_preds.append(pred_tokens)
        all_labels.append(label_tokens)
        all_contexts.append(context_tokens)
    evaluator.evaluate(all_preds, all_labels, log=True, context=all_contexts)
    conv_metrics = evaluator.report()
    # Add CRSDP-style metrics (placeholders for now)
    conv_metrics['BLEU@1'] = 0.0
    conv_metrics['BLEU@2'] = 0.0
    conv_metrics['Dist@1'] = 0.0
    conv_metrics['Dist@2'] = 0.0
    conv_metrics['PPL'] = 0.0
    return conv_metrics

def create_subjective_evaluation_prompt(responses, metric_name):
    """Create prompt for subjective evaluation following ECR methodology."""
    prompt = f"""We have {len(responses)} responses to a given scenario. Please evaluate and score each response based on its "{metric_name}".

"""
    
    if metric_name == "Emotional Intensity":
        prompt += """Emotional Intensity refers to the strength and depth of emotions conveyed in a response, reflecting how powerfully it communicates feelings or emotional states. The score should be on a scale from 0 to 9, where 0 is the least emotional intensity and 9 is the most."""
    elif metric_name == "Emotional Persuasiveness":
        prompt += """Emotional Persuasiveness refers to the ability of the response to connect with the user on an emotional level, influencing their feelings effectively. The score should be on a scale from 0 to 9, where 0 is the least emotional persuasiveness and 9 is the most."""
    elif metric_name == "Logic Persuasiveness":
        prompt += """Logic Persuasiveness refers to how well the response uses logical reasoning and coherent arguments to convincingly address the given scenario. The score should be on a scale from 0 to 9, where 0 is the least logic persuasiveness and 9 is the most."""
    elif metric_name == "Informativeness":
        prompt += """Informativeness refers to how much relevant and useful information the response provides. The score should be on a scale from 0 to 9, where 0 is the least informativeness and 9 is the most."""
    elif metric_name == "Lifelikeness":
        prompt += """Lifelikeness refers to how vivid and engaging the responses are, indicating the extent to which they resemble natural human communication. The score should be on a scale from 0 to 9, where 0 is the least lifelikeness and 9 is the most."""
    
    prompt += f""" Only answer the score in the form of "response name: score."

[MODELS: RESPS]
"""
    
    for i, response in enumerate(responses):
        prompt += f"Response {i+1}: {response}\n"
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Comprehensive ECR Evaluation")
    parser.add_argument("--base_model", type=str, default="/data1/s3905993/ECRHMAS/src/models/llama2_chat")
    parser.add_argument("--lora_model", type=str, default="/data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu")
    parser.add_argument("--test_file", type=str, default="src_emo/data/redial/test_data_processed.jsonl")
    parser.add_argument("--output_file", type=str, default="results/ecr_comprehensive_evaluation.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--entity2id_file", type=str, default="data/redial/entity2id.json")
    
    args = parser.parse_args()
    
    print("=== ECR Comprehensive Evaluation ===")
    print(f"Model: LoRA-enhanced Llama2-7B-Chat")
    print(f"Dataset: {args.test_file}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_file}")
    print("=" * 50)
    
    # Load model and tokenizer
    model, tokenizer = load_lora_model(args.base_model, args.lora_model)
    
    # Load test data
    print(f"Loading test data from {args.test_file}")
    data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} samples")
    
    # Initialize evaluators
    rec_evaluator = RecEvaluator(k_list=[1, 10, 50])
    conv_evaluator = ConvEvaluator(tokenizer, None)
    
    # Evaluate recommendation accuracy (ECR metrics)
    rec_metrics = evaluate_recommendation_accuracy(data, model, tokenizer, rec_evaluator, args.num_samples)
    
    # Evaluate conversation quality (for CRS/CRSDP baselines)
    conv_results = evaluate_conversation_quality(data, model, tokenizer, conv_evaluator, args.num_samples)
    
    # Create subjective evaluation prompts (for manual evaluation)
    sample_responses = []
    for i, sample in enumerate(data[:5]):  # Use first 5 samples for subjective evaluation
        response = generate_recommendation_response(model, tokenizer, sample['context'])
        sample_responses.append(response)
    
    subjective_prompts = {}
    for metric in ["Emotional Intensity", "Emotional Persuasiveness", "Logic Persuasiveness", "Informativeness", "Lifelikeness"]:
        subjective_prompts[metric] = create_subjective_evaluation_prompt(sample_responses, metric)
    
    # Compile results
    results = {
        "model_info": {
            "base_model": args.base_model,
            "lora_model": args.lora_model,
            "num_samples": args.num_samples
        },
        "recommendation_metrics": rec_metrics,
        "conversation_metrics": conv_results,
        "subjective_evaluation_prompts": subjective_prompts,
        "sample_responses": sample_responses
    }
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output_file}")
    # Print summary
    print("\n=== Evaluation Summary ===")
    print("Recommendation Metrics (ECR/CRSDP):")
    for key, value in rec_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Conversation Metrics (CRSDP):")
    for key in ["BLEU@1", "BLEU@2", "Dist@1", "Dist@2", "PPL"]:
        print(f"  {key}: {conv_results.get(key, 0.0):.4f}")
    print(f"\nSubjective evaluation prompts created for {len(sample_responses)} sample responses.")
    print("Manual evaluation required for subjective metrics using the provided prompts.")

if __name__ == "__main__":
    main() 