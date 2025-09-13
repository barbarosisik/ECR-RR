#!/usr/bin/env python3
"""
Mixtral-Based Subjective Evaluation for ECR System
Simplified version using local Mixtral model for offline evaluation
"""

import json
import random
import time
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MixtralSubjectiveEvaluator:
    def __init__(self, model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """Initialize the Mixtral-based subjective evaluator"""
        self.model_path = model_path
        
        print(f"Loading Mixtral model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Simplified evaluation prompts for Mixtral
        self.evaluation_prompts = {
            "emotional_intensity": """<s>[INST] Rate the emotional intensity of this movie recommendation response (0-9, where 0=no emotion, 9=very strong emotion):

Response: "{response}"

Consider: Does it express personal feelings, use emotional words, or convey enthusiasm?

Score (0-9): [/INST]""",
            
            "emotional_persuasiveness": """<s>[INST] Rate the emotional persuasiveness of this movie recommendation response (0-9, where 0=not persuasive, 9=highly persuasive):

Response: "{response}"

Consider: Does it create emotional connection, use emotional appeals, or engage the user?

Score (0-9): [/INST]""",
            
            "logic_persuasiveness": """<s>[INST] Rate the logic persuasiveness of this movie recommendation response (0-9, where 0=not logical, 9=highly logical):

Response: "{response}"

Consider: Does it provide logical reasons, coherent arguments, or factual support?

Score (0-9): [/INST]""",
            
            "informativeness": """<s>[INST] Rate the informativeness of this movie recommendation response (0-9, where 0=not informative, 9=highly informative):

Response: "{response}"

Consider: Does it provide useful movie information, relevant details, or help decision-making?

Score (0-9): [/INST]""",
            
            "lifelikeness": """<s>[INST] Rate the lifelikeness of this movie recommendation response (0-9, where 0=robotic, 9=very human-like):

Response: "{response}"

Consider: Does it sound natural, conversational, and avoid robotic language?

Score (0-9): [/INST]"""
        }
    
    def score_response(self, response: str, dimension: str) -> float:
        """Score a response using Mixtral model"""
        prompt = self.evaluation_prompts[dimension].format(response=response)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            score_text = generated_text[len(prompt):].strip()
            
            # Extract numerical score
            try:
                # Look for numbers in the response
                import re
                numbers = re.findall(r'\d+', score_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0, min(9, score))  # Clamp to 0-9 range
                else:
                    print(f"Warning: No number found in '{score_text}', using 5.0 as default")
                    return 5.0
            except ValueError:
                print(f"Warning: Could not parse score '{score_text}', using 5.0 as default")
                return 5.0
                
        except Exception as e:
            print(f"Error scoring with Mixtral: {e}")
            return 5.0
    
    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate a single response on all 5 dimensions"""
        scores = {}
        
        for dimension in self.evaluation_prompts.keys():
            score = self.score_response(response, dimension)
            scores[dimension] = score
            time.sleep(0.1)  # Small delay
        
        return scores
    
    def evaluate_dataset(self, data: List[Dict[str, Any]], sample_size: int = 100) -> Dict[str, Any]:
        """Evaluate a dataset of responses"""
        if sample_size and len(data) > sample_size:
            data = random.sample(data, sample_size)
        
        print(f"Evaluating {len(data)} responses with Mixtral...")
        
        all_scores = {
            "emotional_intensity": [],
            "emotional_persuasiveness": [],
            "logic_persuasiveness": [],
            "informativeness": [],
            "lifelikeness": []
        }
        
        results = []
        
        for i, item in enumerate(data):
            response = item.get('pred', '')
            if not response.strip():
                continue
            
            print(f"Evaluating response {i+1}/{len(data)}")
            scores = self.evaluate_response(response)
            
            result = {
                'input': item.get('input', ''),
                'response': response,
                'label': item.get('label', ''),
                'scores': scores
            }
            results.append(result)
            
            # Collect scores for averaging
            for dimension, score in scores.items():
                all_scores[dimension].append(score)
        
        # Calculate averages
        averages = {}
        for dimension, scores_list in all_scores.items():
            if scores_list:
                averages[dimension] = sum(scores_list) / len(scores_list)
            else:
                averages[dimension] = 0.0
        
        return {
            'results': results,
            'averages': averages,
            'total_evaluated': len(results)
        }

def load_ecr_results(file_path: str) -> List[Dict[str, Any]]:
    """Load ECR results from JSONL file"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

def save_evaluation_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def print_evaluation_summary(results: Dict[str, Any]):
    """Print evaluation summary in table format"""
    averages = results['averages']
    
    print("\n" + "="*80)
    print("MIXTRAL-BASED SUBJECTIVE EVALUATION RESULTS")
    print("="*80)
    print(f"Total responses evaluated: {results['total_evaluated']}")
    print("\nAverage Scores (0-9 scale):")
    print("-" * 50)
    print(f"Emotional Intensity (Emo Int):     {averages['emotional_intensity']:.3f}")
    print(f"Emotional Persuasiveness (Emo Pers): {averages['emotional_persuasiveness']:.3f}")
    print(f"Logic Persuasiveness (Log Pers):   {averages['logic_persuasiveness']:.3f}")
    print(f"Informativeness (Info):            {averages['informativeness']:.3f}")
    print(f"Lifelikeness (Life):               {averages['lifelikeness']:.3f}")
    print("-" * 50)
    
    # Calculate overall average
    overall_avg = sum(averages.values()) / len(averages)
    print(f"Overall Average:                   {overall_avg:.3f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Mixtral-based subjective evaluation for ECR system")
    parser.add_argument("--input_file", default="save/redial_gen/emp_test.jsonl",
                       help="Path to ECR results file")
    parser.add_argument("--output_file", default="mixtral_evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--model_path", default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                       help="Mixtral model path")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="Number of responses to evaluate (0 for all)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    
    # Initialize evaluator
    evaluator = MixtralSubjectiveEvaluator(args.model_path)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_ecr_results(args.input_file)
    print(f"Loaded {len(data)} responses")
    
    # Evaluate
    results = evaluator.evaluate_dataset(data, args.sample_size)
    
    # Save results
    save_evaluation_results(results, args.output_file)
    print(f"Results saved to {args.output_file}")
    
    # Print summary
    print_evaluation_summary(results)
    
    # Show some example evaluations
    print("\nExample Evaluations:")
    print("-" * 80)
    for i, result in enumerate(results['results'][:3]):
        print(f"\nExample {i+1}:")
        print(f"Response: {result['response'][:200]}...")
        print(f"Scores: Emo Int={result['scores']['emotional_intensity']:.1f}, "
              f"Emo Pers={result['scores']['emotional_persuasiveness']:.1f}, "
              f"Log Pers={result['scores']['logic_persuasiveness']:.1f}, "
              f"Info={result['scores']['informativeness']:.1f}, "
              f"Life={result['scores']['lifelikeness']:.1f}")

if __name__ == "__main__":
    main() 