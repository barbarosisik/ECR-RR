#!/usr/bin/env python3
"""
Simple evaluation script to test the trained enhanced ECR model.
"""

import torch
import json
import logging
from rl.enhanced_critic import EnhancedCriticAgent
from train_ecr_with_enhanced_critic import ECRPolicy
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trained_model(checkpoint_path: str):
    """Test the trained model with sample data."""
    print("🧪 Testing Trained Enhanced ECR Model")
    print("=" * 50)
    
    try:
        # Load trained model
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize models
        policy = ECRPolicy()
        critic = EnhancedCriticAgent()
        
        # Load trained weights
        policy.load_state_dict(checkpoint['policy_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        
        print("✅ Models loaded successfully!")
        
        # Test with sample data
        test_samples = [
            {
                "context": "User: I love action movies with great special effects. Recommender: What about superhero movies?",
                "user_preference": ["Avengers: Endgame", "Iron Man"],
                "candidate_items": ["Avengers: Endgame", "Iron Man", "Black Panther", "Thor: Ragnarok"]
            },
            {
                "context": "User: I'm looking for romantic comedies. Recommender: Have you seen any recent ones?",
                "user_preference": ["La La Land", "The Notebook"],
                "candidate_items": ["La La Land", "The Notebook", "500 Days of Summer", "Crazy Rich Asians"]
            }
        ]
        
        print("\n📊 Testing Model Performance:")
        print("-" * 40)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nSample {i}:")
            print(f"Context: {sample['context']}")
            print(f"User Preference: {sample['user_preference']}")
            
            # Generate response
            response = policy.generate_response(sample['context'])
            print(f"Generated Response: {response}")
            
            # Get recommendations
            recommended_items = policy.get_recommendations(
                sample['context'], 
                response, 
                sample['candidate_items']
            )
            print(f"Recommended Items: {recommended_items}")
            
            # Calculate comprehensive reward
            total_reward, detailed_scores = critic.get_comprehensive_reward(
                context=sample['context'],
                response=response,
                recommended_items=recommended_items,
                user_preference=sample['user_preference']
            )
            
            print(f"Comprehensive Reward: {total_reward:.3f}")
            print(f"Detailed Scores:")
            for dimension, score in detailed_scores.items():
                print(f"  {dimension}: {score:.3f}")
            
            # Calculate NDCG
            ndcg_score = critic.calculate_ndcg(recommended_items, sample['user_preference'], k=4)
            print(f"NDCG@4: {ndcg_score:.3f}")
        
        print("\n🎉 Model evaluation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Trained Enhanced ECR Model")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="ecr_enhanced_critic_output/checkpoint_epoch_4.pt",
                       help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    # Test the trained model
    success = test_trained_model(args.checkpoint_path)
    
    if success:
        print("\n✅ Model testing successful!")
    else:
        print("\n❌ Model testing failed!")

if __name__ == "__main__":
    main() 