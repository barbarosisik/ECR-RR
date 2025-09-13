import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainedCriticRobertaMultiHead(nn.Module):
    """Trained RoBERTa critic model with 5 output heads."""
    
    def __init__(self, model_name="roberta-base", num_outputs=5):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.roberta.config.hidden_size, num_outputs)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)

class EnhancedCriticAgentWithTrainedModel(nn.Module):
    """
    Enhanced Critic Agent that uses our trained RoBERTa model for subjective evaluation
    combined with NDCG-based recommendation accuracy.
    
    This uses our trained critic model for:
    1. Subjective evaluation (empathy, persuasiveness, logic, informativeness, lifelikeness)
    2. Recommendation accuracy using NDCG@k
    """
    
    def __init__(self, trained_model_path: str = "critic_roberta_best.pth", num_dimensions: int = 6):
        super().__init__()
        self.num_dimensions = num_dimensions
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Load our trained model
        self.trained_critic = TrainedCriticRobertaMultiHead()
        if os.path.exists(trained_model_path):
            state_dict = torch.load(trained_model_path, map_location='cpu')
            self.trained_critic.load_state_dict(state_dict)
            logger.info(f"Loaded trained critic model from {trained_model_path}")
        else:
            logger.warning(f"Trained model not found at {trained_model_path}, using untrained model")
        
        # Recommendation accuracy head (for NDCG calculation)
        self.recommendation_head = nn.Linear(768, 1)  # RoBERTa hidden size is 768
        
        self.dropout = nn.Dropout(0.1)
        
        # Default weights for comprehensive reward
        self.default_weights = {
            'empathy': 0.25,
            'persuasiveness': 0.15,
            'logic': 0.15,
            'informativeness': 0.10,
            'lifelikeness': 0.10,
            'recommendation_accuracy': 0.25
        }
        
        # Map output indices to dimensions
        self.dimension_mapping = {
            0: 'empathy',
            1: 'persuasiveness', 
            2: 'logic',
            3: 'informativeness',
            4: 'lifelikeness'
        }
        
        logger.info(f"Enhanced Critic Agent with trained model initialized")
    
    def forward(self, context: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get subjective evaluation scores using trained model.
        
        Args:
            context: Dialogue context
            response: Generated response
            
        Returns:
            Dictionary of subjective scores
        """
        # Get device
        device = next(self.parameters()).device
        
        # Prepare input (same format as training)
        input_text = f"<context>{context}</context><response>{response}</response>"
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256,
            padding=True
        )
        
        # Move inputs to device
        inputs.input_ids = inputs.input_ids.to(device)
        inputs.attention_mask = inputs.attention_mask.to(device)
        
        # Get trained model predictions
        with torch.no_grad():
            self.trained_critic = self.trained_critic.to(device)
            outputs = self.trained_critic(inputs.input_ids, inputs.attention_mask)
            scores = outputs.squeeze()
        
        # Map scores to dimensions
        subjective_scores = {}
        for i, dimension in self.dimension_mapping.items():
            subjective_scores[dimension] = scores[i].unsqueeze(0)
        
        return subjective_scores
    
    def calculate_ndcg(self, recommended_items: List[str], user_preference: List[str], k: int = 10) -> float:
        """
        Calculate NDCG@k for recommendation accuracy.
        
        Args:
            recommended_items: List of recommended item IDs
            user_preference: List of user's preferred item IDs
            k: Number of top items to consider
            
        Returns:
            NDCG@k score
        """
        if not recommended_items or not user_preference:
            return 0.0
        
        # Create relevance scores (1 if item is in user preference, 0 otherwise)
        relevance = []
        for item in recommended_items[:k]:
            relevance.append(1.0 if item in user_preference else 0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(user_preference), k)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def get_recommendation_accuracy_score(self, context: str, response: str, 
                                        recommended_items: List[str], 
                                        user_preference: List[str], 
                                        k: int = 10) -> float:
        """
        Get recommendation accuracy score using NDCG.
        
        Args:
            context: Dialogue context
            response: Generated response
            recommended_items: List of recommended item IDs
            user_preference: List of user's preferred item IDs
            k: Number of top items to consider
            
        Returns:
            NDCG@k score
        """
        return self.calculate_ndcg(recommended_items, user_preference, k)
    
    def get_comprehensive_reward(self, context: str, response: str, 
                               recommended_items: List[str], 
                               user_preference: List[str],
                               weights: Optional[Dict[str, float]] = None,
                               k: int = 10) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward combining subjective evaluation and recommendation accuracy.
        
        Args:
            context: Dialogue context
            response: Generated response
            recommended_items: List of recommended item IDs
            user_preference: List of user's preferred item IDs
            weights: Optional weights for different components
            k: Number of top items to consider for NDCG
            
        Returns:
            Tuple of (total_reward, component_scores)
        """
        if weights is None:
            weights = self.default_weights
        
        # Get subjective scores
        subjective_scores = self.forward(context, response)
        
        # Get recommendation accuracy
        recommendation_score = self.get_recommendation_accuracy_score(
            context, response, recommended_items, user_preference, k
        )
        
        # Calculate weighted reward
        total_reward = 0.0
        component_scores = {}
        
        # Add subjective scores
        for dimension, score in subjective_scores.items():
            if dimension in weights:
                weighted_score = score.item() * weights[dimension]
                total_reward += weighted_score
                component_scores[dimension] = score.item()
        
        # Add recommendation accuracy
        if 'recommendation_accuracy' in weights:
            weighted_rec_score = recommendation_score * weights['recommendation_accuracy']
            total_reward += weighted_rec_score
            component_scores['recommendation_accuracy'] = recommendation_score
        
        return total_reward, component_scores
    
    def get_subjective_reward(self, context: str, response: str, 
                            weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Get subjective evaluation reward only.
        
        Args:
            context: Dialogue context
            response: Generated response
            weights: Optional weights for different dimensions
            
        Returns:
            Tuple of (total_reward, component_scores)
        """
        if weights is None:
            weights = {k: v for k, v in self.default_weights.items() if k != 'recommendation_accuracy'}
        
        subjective_scores = self.forward(context, response)
        
        total_reward = 0.0
        component_scores = {}
        
        for dimension, score in subjective_scores.items():
            if dimension in weights:
                weighted_score = score.item() * weights[dimension]
                total_reward += weighted_score
                component_scores[dimension] = score.item()
        
        return total_reward, component_scores
    
    def save_model(self, path: str):
        """Save the enhanced critic model."""
        torch.save({
            'trained_critic_state_dict': self.trained_critic.state_dict(),
            'recommendation_head_state_dict': self.recommendation_head.state_dict(),
            'default_weights': self.default_weights
        }, path)
        logger.info(f"Enhanced critic model saved to {path}")
    
    def load_model(self, path: str):
        """Load the enhanced critic model."""
        checkpoint = torch.load(path, map_location='cpu')
        self.trained_critic.load_state_dict(checkpoint['trained_critic_state_dict'])
        self.recommendation_head.load_state_dict(checkpoint['recommendation_head_state_dict'])
        self.default_weights = checkpoint.get('default_weights', self.default_weights)
        logger.info(f"Enhanced critic model loaded from {path}")

def test_enhanced_critic_with_trained_model():
    """Test the enhanced critic with trained model."""
    print("Testing Enhanced Critic with Trained Model...")
    
    # Initialize critic
    critic = EnhancedCriticAgentWithTrainedModel()
    
    # Test data
    context = "Hi, I love action movies. What would you recommend?"
    response = "That's great! I think you'd really enjoy 'Mad Max: Fury Road' - it's an incredible action film with amazing stunts and a compelling story. Have you seen it?"
    recommended_items = ["movie_123", "movie_456", "movie_789"]
    user_preference = ["movie_123", "movie_789"]
    
    # Test subjective evaluation
    print("\n1. Testing subjective evaluation...")
    subjective_scores = critic.forward(context, response)
    for dimension, score in subjective_scores.items():
        print(f"   {dimension}: {score.item():.3f}")
    
    # Test recommendation accuracy
    print("\n2. Testing recommendation accuracy...")
    rec_score = critic.get_recommendation_accuracy_score(context, response, recommended_items, user_preference)
    print(f"   NDCG@10: {rec_score:.3f}")
    
    # Test comprehensive reward
    print("\n3. Testing comprehensive reward...")
    total_reward, components = critic.get_comprehensive_reward(context, response, recommended_items, user_preference)
    print(f"   Total reward: {total_reward:.3f}")
    print("   Components:")
    for component, score in components.items():
        print(f"     {component}: {score:.3f}")
    
    print("\n✅ Enhanced critic with trained model test completed!")

if __name__ == "__main__":
    test_enhanced_critic_with_trained_model() 