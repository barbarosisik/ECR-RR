#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Enhanced ECR
Compares enhanced ECR (with NDCG rewards) against baseline ECR system.
"""

import torch
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rl.enhanced_critic import EnhancedCriticAgent
from train_ecr_with_enhanced_critic import ECRPolicy
import argparse
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedECREvaluator:
    """Evaluator for Enhanced ECR system."""
    
    def __init__(self, policy_model_path: str = None, critic_model_path: str = None):
        self.critic = EnhancedCriticAgent()
        self.policy = ECRPolicy()
        
        # Load trained models if provided
        if policy_model_path and os.path.exists(policy_model_path):
            self.load_policy(policy_model_path)
        if critic_model_path and os.path.exists(critic_model_path):
            self.critic.load_model(critic_model_path)
        
        logger.info("Enhanced ECR Evaluator initialized")
    
    def load_policy(self, model_path: str):
        """Load trained policy model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f"Policy model loaded from {model_path}")
    
    def evaluate_single_sample(self, context: str, user_preference: List[str], 
                             candidate_items: List[str], ground_truth_response: str = None) -> Dict:
        """Evaluate a single dialogue sample."""
        # Generate response using policy
        response = self.policy.generate_response(context)
        
        # Get recommendations
        recommended_items = self.policy.get_recommendations(context, response, candidate_items)
        
        # Calculate comprehensive reward
        total_reward, detailed_scores = self.critic.get_comprehensive_reward(
            context=context,
            response=response,
            recommended_items=recommended_items,
            user_preference=user_preference
        )
        
        # Calculate subjective-only reward
        subjective_reward, subjective_scores = self.critic.get_subjective_reward(
            context=context,
            response=response
        )
        
        # Calculate NDCG
        ndcg_score = self.critic.calculate_ndcg(recommended_items, user_preference, k=10)
        
        return {
            'context': context,
            'generated_response': response,
            'ground_truth_response': ground_truth_response,
            'recommended_items': recommended_items,
            'user_preference': user_preference,
            'total_reward': total_reward,
            'subjective_reward': subjective_reward,
            'ndcg_score': ndcg_score,
            'detailed_scores': detailed_scores,
            'subjective_scores': subjective_scores
        }
    
    def evaluate_dataset(self, data_path: str, num_samples: int = 100) -> Dict:
        """Evaluate on a dataset."""
        logger.info(f"Evaluating on dataset: {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
        # Sample data if needed
        if num_samples and len(data) > num_samples:
            data = np.random.choice(data, num_samples, replace=False).tolist()
        
        results = []
        
        for i, item in enumerate(tqdm(data, desc="Evaluating samples")):
            try:
                # Extract context and preferences
                context = self.extract_context(item)
                user_preference = self.extract_user_preference(item)
                candidate_items = item.get('candidate_items', [])
                ground_truth_response = item.get('response', '')
                
                if context and user_preference:
                    result = self.evaluate_single_sample(
                        context=context,
                        user_preference=user_preference,
                        candidate_items=candidate_items,
                        ground_truth_response=ground_truth_response
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        return self.compute_metrics(results)
    
    def extract_context(self, item: Dict) -> str:
        """Extract dialogue context from data item."""
        if 'conversation' in item:
            conversation = item['conversation']
            if len(conversation) >= 2:
                context_turns = conversation[-2:]
                context = " ".join([turn.get('text', '') for turn in context_turns])
                return context
        return ""
    
    def extract_user_preference(self, item: Dict) -> List[str]:
        """Extract user preferences from data item."""
        if 'user_preference' in item:
            return item['user_preference']
        elif 'liked_items' in item:
            return item['liked_items']
        return []
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute comprehensive evaluation metrics."""
        if not results:
            return {}
        
        # Extract metrics
        total_rewards = [r['total_reward'] for r in results]
        subjective_rewards = [r['subjective_reward'] for r in results]
        ndcg_scores = [r['ndcg_score'] for r in results]
        
        # Detailed scores
        detailed_metrics = {
            'empathy': [], 'persuasiveness': [], 'logic': [], 
            'informativeness': [], 'lifelikeness': [], 'recommendation_accuracy': []
        }
        
        for result in results:
            for dimension, score in result['detailed_scores'].items():
                if dimension in detailed_metrics:
                    detailed_metrics[dimension].append(score)
        
        # Compute statistics
        metrics = {
            'num_samples': len(results),
            'total_reward': {
                'mean': np.mean(total_rewards),
                'std': np.std(total_rewards),
                'min': np.min(total_rewards),
                'max': np.max(total_rewards)
            },
            'subjective_reward': {
                'mean': np.mean(subjective_rewards),
                'std': np.std(subjective_rewards),
                'min': np.min(subjective_rewards),
                'max': np.max(subjective_rewards)
            },
            'ndcg_score': {
                'mean': np.mean(ndcg_scores),
                'std': np.std(ndcg_scores),
                'min': np.min(ndcg_scores),
                'max': np.max(ndcg_scores)
            },
            'detailed_scores': {}
        }
        
        # Compute detailed score statistics
        for dimension, scores in detailed_metrics.items():
            if scores:
                metrics['detailed_scores'][dimension] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        return metrics
    
    def compare_with_baseline(self, enhanced_results: Dict, baseline_results: Dict) -> Dict:
        """Compare enhanced ECR with baseline."""
        comparison = {
            'improvement': {},
            'relative_improvement': {}
        }
        
        # Compare total reward
        if 'total_reward' in enhanced_results and 'total_reward' in baseline_results:
            enhanced_mean = enhanced_results['total_reward']['mean']
            baseline_mean = baseline_results['total_reward']['mean']
            
            improvement = enhanced_mean - baseline_mean
            relative_improvement = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
            
            comparison['improvement']['total_reward'] = improvement
            comparison['relative_improvement']['total_reward'] = relative_improvement
        
        # Compare NDCG
        if 'ndcg_score' in enhanced_results and 'ndcg_score' in baseline_results:
            enhanced_ndcg = enhanced_results['ndcg_score']['mean']
            baseline_ndcg = baseline_results['ndcg_score']['mean']
            
            improvement = enhanced_ndcg - baseline_ndcg
            relative_improvement = (improvement / baseline_ndcg) * 100 if baseline_ndcg > 0 else 0
            
            comparison['improvement']['ndcg_score'] = improvement
            comparison['relative_improvement']['ndcg_score'] = relative_improvement
        
        # Compare detailed scores
        comparison['detailed_improvements'] = {}
        for dimension in ['empathy', 'persuasiveness', 'logic', 'informativeness', 'lifelikeness']:
            if (dimension in enhanced_results.get('detailed_scores', {}) and 
                dimension in baseline_results.get('detailed_scores', {})):
                
                enhanced_score = enhanced_results['detailed_scores'][dimension]['mean']
                baseline_score = baseline_results['detailed_scores'][dimension]['mean']
                
                improvement = enhanced_score - baseline_score
                relative_improvement = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
                
                comparison['detailed_improvements'][dimension] = {
                    'absolute': improvement,
                    'relative': relative_improvement
                }
        
        return comparison


def create_baseline_evaluator():
    """Create a baseline evaluator (original ECR without RL)."""
    # This would be your original ECR system
    # For now, we'll create a simple baseline
    class BaselineEvaluator:
        def __init__(self):
            self.critic = EnhancedCriticAgent()
            self.policy = ECRPolicy()
        
        def evaluate_single_sample(self, context: str, user_preference: List[str], 
                                 candidate_items: List[str], ground_truth_response: str = None) -> Dict:
            # Use the same evaluation logic but without RL training
            response = self.policy.generate_response(context)
            recommended_items = self.policy.get_recommendations(context, response, candidate_items)
            
            total_reward, detailed_scores = self.critic.get_comprehensive_reward(
                context=context,
                response=response,
                recommended_items=recommended_items,
                user_preference=user_preference
            )
            
            subjective_reward, subjective_scores = self.critic.get_subjective_reward(
                context=context,
                response=response
            )
            
            ndcg_score = self.critic.calculate_ndcg(recommended_items, user_preference, k=10)
            
            return {
                'context': context,
                'generated_response': response,
                'ground_truth_response': ground_truth_response,
                'recommended_items': recommended_items,
                'user_preference': user_preference,
                'total_reward': total_reward,
                'subjective_reward': subjective_reward,
                'ndcg_score': ndcg_score,
                'detailed_scores': detailed_scores,
                'subjective_scores': subjective_scores
            }
        
        def evaluate_dataset(self, data_path: str, num_samples: int = 100) -> Dict:
            # Same evaluation logic as enhanced evaluator
            evaluator = EnhancedECREvaluator()
            return evaluator.evaluate_dataset(data_path, num_samples)
    
    return BaselineEvaluator()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enhanced ECR")
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to evaluation data")
    parser.add_argument("--policy_model_path", type=str, default=None,
                       help="Path to trained policy model")
    parser.add_argument("--critic_model_path", type=str, default=None,
                       help="Path to trained critic model")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Enhanced ECR Evaluation")
    print("=" * 50)
    
    # Evaluate Enhanced ECR
    print("1. Evaluating Enhanced ECR...")
    enhanced_evaluator = EnhancedECREvaluator(
        policy_model_path=args.policy_model_path,
        critic_model_path=args.critic_model_path
    )
    enhanced_results = enhanced_evaluator.evaluate_dataset(args.data_path, args.num_samples)
    
    # Evaluate Baseline
    print("2. Evaluating Baseline ECR...")
    baseline_evaluator = create_baseline_evaluator()
    baseline_results = baseline_evaluator.evaluate_dataset(args.data_path, args.num_samples)
    
    # Compare results
    print("3. Comparing results...")
    comparison = enhanced_evaluator.compare_with_baseline(enhanced_results, baseline_results)
    
    # Print results
    print("\n📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print("\nEnhanced ECR Results:")
    print(f"  Total Reward: {enhanced_results['total_reward']['mean']:.3f} ± {enhanced_results['total_reward']['std']:.3f}")
    print(f"  Subjective Reward: {enhanced_results['subjective_reward']['mean']:.3f} ± {enhanced_results['subjective_reward']['std']:.3f}")
    print(f"  NDCG Score: {enhanced_results['ndcg_score']['mean']:.3f} ± {enhanced_results['ndcg_score']['std']:.3f}")
    
    print("\nBaseline ECR Results:")
    print(f"  Total Reward: {baseline_results['total_reward']['mean']:.3f} ± {baseline_results['total_reward']['std']:.3f}")
    print(f"  Subjective Reward: {baseline_results['subjective_reward']['mean']:.3f} ± {baseline_results['subjective_reward']['std']:.3f}")
    print(f"  NDCG Score: {baseline_results['ndcg_score']['mean']:.3f} ± {baseline_results['ndcg_score']['std']:.3f}")
    
    print("\n🎯 IMPROVEMENTS")
    print("=" * 30)
    
    if 'total_reward' in comparison['improvement']:
        improvement = comparison['improvement']['total_reward']
        relative = comparison['relative_improvement']['total_reward']
        print(f"Total Reward: {improvement:+.3f} ({relative:+.1f}%)")
    
    if 'ndcg_score' in comparison['improvement']:
        improvement = comparison['improvement']['ndcg_score']
        relative = comparison['relative_improvement']['ndcg_score']
        print(f"NDCG Score: {improvement:+.3f} ({relative:+.1f}%)")
    
    print("\nDetailed Improvements:")
    for dimension, improvements in comparison.get('detailed_improvements', {}).items():
        absolute = improvements['absolute']
        relative = improvements['relative']
        print(f"  {dimension}: {absolute:+.3f} ({relative:+.1f}%)")
    
    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'enhanced_results': enhanced_results,
            'baseline_results': baseline_results,
            'comparison': comparison
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main() 