import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCriticAgent(nn.Module):
    """
    Enhanced Critic Agent that combines subjective evaluation with NDCG-based recommendation accuracy.
    
    This extends the original CriticAgent to include:
    1. Subjective evaluation (empathy, persuasiveness, logic, informativeness, lifelikeness)
    2. Recommendation accuracy using NDCG@k
    """
    
    def __init__(self, model_name: str = "roberta-base", num_dimensions: int = 6):
        super().__init__()
        self.model_name = model_name
        self.num_dimensions = num_dimensions
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        # Force model to be on CPU initially to avoid device issues
        self.roberta = self.roberta.cpu()
        
        # Add special tokens for context and response
        special_tokens = {
            'additional_special_tokens': ['<context>', '</context>', '<response>', '</response>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.roberta.resize_token_embeddings(len(self.tokenizer))
        
        # Subjective evaluation heads (5 dimensions)
        self.subjective_heads = nn.ModuleDict({
            'empathy': nn.Linear(self.roberta.config.hidden_size, 1),
            'persuasiveness': nn.Linear(self.roberta.config.hidden_size, 1),
            'logic': nn.Linear(self.roberta.config.hidden_size, 1),
            'informativeness': nn.Linear(self.roberta.config.hidden_size, 1),
            'lifelikeness': nn.Linear(self.roberta.config.hidden_size, 1)
        })
        
        # Recommendation accuracy head (for NDCG calculation)
        self.recommendation_head = nn.Linear(self.roberta.config.hidden_size, 1)
        
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
        
        logger.info(f"Enhanced Critic Agent initialized with {num_dimensions} dimensions")
    
    def forward(self, context: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get subjective evaluation scores.
        
        Args:
            context: Dialogue context
            response: Generated response
            
        Returns:
            Dictionary of subjective scores
        """
        # Get device
        device = next(self.parameters()).device
        
        # Prepare input
        input_text = f"<context>{context}</context><response>{response}</response>"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to device
        inputs.input_ids = inputs.input_ids.to(device)
        inputs.attention_mask = inputs.attention_mask.to(device)
        
        # Get RoBERTa embeddings
        with torch.no_grad():
            # Force move RoBERTa and all its components to device
            self.roberta = self.roberta.to(device)
            # Also move the embedding layer explicitly
            if hasattr(self.roberta, 'embeddings'):
                self.roberta.embeddings = self.roberta.embeddings.to(device)
            outputs = self.roberta(**inputs)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            pooled_output = self.dropout(pooled_output)
        
        # Get subjective scores
        subjective_scores = {}
        for dimension, head in self.subjective_heads.items():
            score = head(pooled_output)
            subjective_scores[dimension] = torch.sigmoid(score) * 9.0  # Scale to 0-9 range
        
        return subjective_scores
    
    def calculate_ndcg(self, recommended_items: List[str], user_preference: List[str], k: int = 10) -> float:
        """
        Calculate NDCG@k for recommendation accuracy.
        
        Args:
            recommended_items: List of recommended item IDs
            user_preference: Ground truth preferred items
            k: Top-k items to consider
            
        Returns:
            NDCG@k score (0-1)
        """
        if not recommended_items or not user_preference:
            return 0.0
        
        # Calculate relevance scores (1 if item is in user preference, 0 otherwise)
        relevance = [1.0 if item in user_preference else 0.0 for item in recommended_items[:k]]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = sum(relevance[i] / np.log2(i + 2) for i in range(len(relevance)))
        
        # Calculate IDCG (Ideal DCG) - perfect ranking
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(len(ideal_relevance)))
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_recommendation_accuracy_score(self, context: str, response: str, 
                                        recommended_items: List[str], 
                                        user_preference: List[str], 
                                        k: int = 10) -> float:
        """
        Get recommendation accuracy score using NDCG.
        
        Args:
            context: Dialogue context
            response: Generated response
            recommended_items: List of recommended items
            user_preference: User's preferred items
            k: Top-k for NDCG calculation
            
        Returns:
            NDCG@k score
        """
        ndcg_score = self.calculate_ndcg(recommended_items, user_preference, k)
        return ndcg_score
    
    def get_comprehensive_reward(self, context: str, response: str, 
                               recommended_items: List[str], 
                               user_preference: List[str],
                               weights: Optional[Dict[str, float]] = None,
                               k: int = 10) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive reward combining subjective and objective metrics.
        
        Args:
            context: Dialogue context
            response: Generated response
            recommended_items: List of recommended items
            user_preference: User's preferred items
            weights: Custom weights for different dimensions
            k: Top-k for NDCG calculation
            
        Returns:
            Tuple of (total_reward, detailed_scores)
        """
        if weights is None:
            weights = self.default_weights.copy()
        
        # Get subjective scores
        subjective_scores = self.forward(context, response)
        
        # Get recommendation accuracy score
        recommendation_score = self.get_recommendation_accuracy_score(
            context, response, recommended_items, user_preference, k
        )
        
        # Calculate weighted reward
        total_reward = 0.0
        detailed_scores = {}
        
        # Add subjective scores
        for dimension, score in subjective_scores.items():
            weight = weights.get(dimension, 0.0)
            total_reward += weight * score.item()
            detailed_scores[dimension] = score.item()
        
        # Add recommendation accuracy
        recommendation_weight = weights.get('recommendation_accuracy', 0.0)
        total_reward += recommendation_weight * recommendation_score
        detailed_scores['recommendation_accuracy'] = recommendation_score
        
        return total_reward, detailed_scores
    
    def get_subjective_reward(self, context: str, response: str, 
                            weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Get only subjective reward (for comparison).
        
        Args:
            context: Dialogue context
            response: Generated response
            weights: Custom weights for different dimensions
            
        Returns:
            Tuple of (subjective_reward, detailed_scores)
        """
        if weights is None:
            weights = {k: v for k, v in self.default_weights.items() if k != 'recommendation_accuracy'}
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
        
        subjective_scores = self.forward(context, response)
        
        subjective_reward = 0.0
        detailed_scores = {}
        
        for dimension, score in subjective_scores.items():
            weight = weights.get(dimension, 0.0)
            subjective_reward += weight * score.item()
            detailed_scores[dimension] = score.item()
        
        return subjective_reward, detailed_scores
    
    def save_model(self, path: str):
        """Save the enhanced critic agent."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name,
            'default_weights': self.default_weights
        }, path)
        logger.info(f"Enhanced Critic Agent saved to {path}")
    
    def load_model(self, path: str):
        """Load the enhanced critic agent."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.default_weights = checkpoint.get('default_weights', self.default_weights)
        logger.info(f"Enhanced Critic Agent loaded from {path}")


def test_enhanced_critic():
    """Test the enhanced critic agent."""
    print("Testing Enhanced Critic Agent...")
    
    # Initialize critic
    critic = EnhancedCriticAgent()
    
    # Test data
    context = "User: I love action movies with great special effects. Recommender: What about superhero movies?"
    response = "I absolutely love Marvel movies! The special effects are incredible and the action sequences are mind-blowing. You should definitely watch Avengers: Endgame - it's epic!"
    recommended_items = ["Avengers: Endgame", "Iron Man", "Black Panther", "Thor: Ragnarok"]
    user_preference = ["Avengers: Endgame", "Iron Man"]
    
    # Test subjective evaluation
    print("\n1. Testing Subjective Evaluation:")
    subjective_reward, subjective_scores = critic.get_subjective_reward(context, response)
    print(f"Subjective Reward: {subjective_reward:.3f}")
    for dimension, score in subjective_scores.items():
        print(f"  {dimension}: {score:.3f}")
    
    # Test NDCG calculation
    print("\n2. Testing NDCG Calculation:")
    ndcg_score = critic.calculate_ndcg(recommended_items, user_preference, k=4)
    print(f"NDCG@4: {ndcg_score:.3f}")
    
    # Test comprehensive reward
    print("\n3. Testing Comprehensive Reward:")
    total_reward, detailed_scores = critic.get_comprehensive_reward(
        context, response, recommended_items, user_preference
    )
    print(f"Total Reward: {total_reward:.3f}")
    for dimension, score in detailed_scores.items():
        print(f"  {dimension}: {score:.3f}")
    
    print("\n✅ Enhanced Critic Agent test completed successfully!")


if __name__ == "__main__":
    test_enhanced_critic() 