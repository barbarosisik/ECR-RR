"""
Dialogue Policy Learning Module for ECR-main
Based on CRSDP paper: Knowledge-Based Conversational Recommender Systems Enhanced by Dialogue Policy Learning
Implements actor-critic framework with PPO for dialogue policy optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DialogueAction(Enum):
    """Dialogue actions as defined in CRSDP paper"""
    REQUEST = "request"      # Ask for user preference
    RESPOND = "respond"      # Respond to user feedback
    RECOMMEND = "recommend"  # Provide recommendation
    EXPLAIN = "explain"      # Explain recommendation
    OTHERS = "others"        # Greetings, farewells, chitchat

@dataclass
class DialogueState:
    """Dialogue state representation as per CRSDP paper"""
    user_intention: torch.Tensor  # Latest user intention embedding
    system_action: torch.Tensor   # Latest system action embedding  
    context_embedding: torch.Tensor  # Context embedding from BERT
    dialogue_history: List[str]   # Previous dialogue turns
    user_profile: torch.Tensor    # User preference embedding
    turn_count: int               # Current turn number

class DialoguePolicyNetwork(nn.Module):
    """
    Neural Policy Network for dialogue action selection
    Based on CRSDP paper's actor-critic framework
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,  # 5 dialogue actions
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # State encoder (combines all state components)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network
        Returns: (action_logits, state_value)
        """
        encoded_state = self.state_encoder(state)
        action_logits = self.actor(encoded_state)
        state_value = self.critic(encoded_state)
        
        return action_logits, state_value
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using policy
        Returns: (action_idx, action_prob, state_value)
        """
        action_logits, state_value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if training:
            # Sample action during training
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            action_prob = action_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        else:
            # Greedy action selection during inference
            action_idx = torch.argmax(action_probs, dim=-1)
            action_prob = torch.max(action_probs, dim=-1)[0]
        
        return action_idx, action_prob, state_value.squeeze(-1)

class UserSimulator:
    """
    User Simulator for RL Environment
    Based on CRSDP paper's environment design
    """
    
    def __init__(self, desired_items: List[str], max_turns: int = 20):
        self.desired_items = desired_items
        self.max_turns = max_turns
        self.current_turn = 0
        self.dialogue_history = []
        self.user_profile = []
        
    def reset(self) -> Dict:
        """Reset simulator for new dialogue"""
        self.current_turn = 0
        self.dialogue_history = []
        self.user_profile = []
        return self._get_state()
    
    def step(self, system_action: DialogueAction, system_response: str, recommended_item: Optional[str] = None) -> Tuple[Dict, float, bool]:
        """
        Take a step in the dialogue
        Returns: (next_state, reward, done)
        """
        self.current_turn += 1
        self.dialogue_history.append({
            'turn': self.current_turn,
            'action': system_action.value,
            'response': system_response,
            'recommended_item': recommended_item
        })
        
        # Calculate reward
        reward = self._calculate_reward(system_action, recommended_item)
        
        # Check if dialogue should end
        done = self._is_dialogue_done(recommended_item)
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self, action: DialogueAction, recommended_item: Optional[str], response_quality: Optional[Dict] = None) -> float:
        """Calculate reward based on ECR emotional-empathical responsiveness metrics + NDCG"""
        reward = 0.0
        
        # NDCG-based reward for recommendation quality (from CRSDP)
        if recommended_item and recommended_item in self.desired_items:
            ndcg_reward = self._calculate_ndcg(recommended_item)
            reward += 1.0 * ndcg_reward  # NDCG component
        
        # ECR Emotional-Empathical Responsiveness Metrics (6 heads)
        if response_quality:
            # Emotional Intensity: How emotionally intense is the response? (1-10 scale)
            emotional_intensity = response_quality.get('emotional_intensity', 5.0) / 10.0
            reward += 0.3 * emotional_intensity
            
            # Emotional Persuasiveness: How emotionally persuasive is the response? (1-10 scale)
            emotional_persuasiveness = response_quality.get('emotional_persuasiveness', 5.0) / 10.0
            reward += 0.4 * emotional_persuasiveness
            
            # Logic Persuasiveness: How logically persuasive is the response? (1-10 scale)
            logic_persuasiveness = response_quality.get('logic_persuasiveness', 5.0) / 10.0
            reward += 0.3 * logic_persuasiveness
            
            # Informativeness: How informative is the response? (1-10 scale)
            informativeness = response_quality.get('informativeness', 5.0) / 10.0
            reward += 0.2 * informativeness
            
            # Lifelikeness: How natural and human-like is the response? (1-10 scale)
            lifelikeness = response_quality.get('lifelikeness', 5.0) / 10.0
            reward += 0.3 * lifelikeness
        
        # Negative reward for too many turns
        if self.current_turn > self.max_turns:
            reward -= 2.0
        
        # Small negative reward per turn to encourage efficiency
        reward -= 0.1
        
        # Action-specific rewards
        if action == DialogueAction.RECOMMEND:
            if recommended_item and recommended_item in self.desired_items:
                reward += 0.5
            else:
                reward -= 0.3
        elif action == DialogueAction.EXPLAIN:
            # Reward for explanation actions (empathy-focused)
            reward += 0.2
        elif action == DialogueAction.RESPOND:
            # Reward for empathetic responses
            reward += 0.1
        
        return reward
    
    def _calculate_ndcg(self, recommended_item: str) -> float:
        """Calculate NDCG score for recommendation quality"""
        if recommended_item in self.desired_items:
            # Simple NDCG calculation
            rank = self.desired_items.index(recommended_item)
            return 1.0 / np.log2(rank + 2)  # +2 to avoid log(1) = 0
        return 0.0
    
    def _calculate_ecr_metrics(self, response: str) -> Dict[str, float]:
        """Calculate ECR emotional-empathical responsiveness metrics"""
        # This would ideally use our trained critic model
        # For now, return placeholder values
        return {
            'emotional_intensity': 5.0,
            'emotional_persuasiveness': 5.0,
            'logic_persuasiveness': 5.0,
            'informativeness': 5.0,
            'lifelikeness': 5.0
        }
    
    def _is_dialogue_done(self, recommended_item: Optional[str]) -> bool:
        """Check if dialogue should end"""
        # End if successful recommendation
        if recommended_item and recommended_item in self.desired_items:
            return True
        
        # End if too many turns
        if self.current_turn >= self.max_turns:
            return True
        
        return False
    
    def _get_state(self) -> Dict:
        """Get current state representation"""
        return {
            'turn_count': self.current_turn,
            'dialogue_history': self.dialogue_history,
            'user_profile': self.user_profile,
            'desired_items': self.desired_items
        }

class DialoguePolicyTrainer:
    """
    Dialogue Policy Trainer using PPO
    Based on CRSDP paper's training methodology
    """
    
    def __init__(
        self,
        policy_network: DialoguePolicyNetwork,
        config: Dict,
        device: str = "cuda"
    ):
        self.policy_network = policy_network
        self.config = config
        self.device = device
        
        # PPO hyperparameters
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.ppo_clip_epsilon = config.get('ppo_clip_epsilon', 0.2)
        self.ppo_entropy_coef = config.get('ppo_entropy_coef', 0.01)
        self.ppo_value_coef = config.get('ppo_value_coef', 0.5)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_network.parameters(),
            lr=config.get('rl_learning_rate', 1e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single PPO training step
        Based on CRSDP paper's PPO implementation
        """
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        for _ in range(self.ppo_epochs):
            # Forward pass
            action_logits, state_values = self.policy_network(states)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(-1), returns)
            
            # Entropy bonus for exploration
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            loss = (
                policy_loss + 
                self.ppo_value_coef * value_loss + 
                self.ppo_entropy_coef * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.get('ppo_max_grad_norm', 0.5))
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'total_loss': total_loss / self.ppo_epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def compute_advantages(
        self,
        rewards: List[float],
        state_values: List[float],
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        """
        advantages = []
        returns = []
        
        # Compute returns
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        state_values = torch.tensor(state_values, dtype=torch.float32, device=self.device)
        
        # Compute advantages using GAE
        advantages = returns - state_values
        
        return advantages, returns
    
    def save_checkpoint(self, path: str):
        """Save policy network checkpoint"""
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load policy network checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step'] 