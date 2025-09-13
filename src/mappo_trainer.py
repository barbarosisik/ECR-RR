"""
Multi-Agent PPO (MAPPO) Trainer for ECR

This module implements a multi-agent PPO trainer inspired by the MACPO paper,
enabling multiple agents to learn cooperatively while maintaining individual specializations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

from .ppo_trainer import PPOTrainer
from .critic import CriticAgent
from .reward_functions import RewardCalculator

@dataclass
class AgentConfig:
    """Configuration for individual agents in the multi-agent system."""
    name: str
    model_path: str
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    specialization: str = "general"  # "empathy", "recommendation", "conversation"

class MultiAgentPPOTrainer:
    """
    Multi-Agent PPO Trainer inspired by MACPO.
    
    This trainer enables multiple agents to learn cooperatively:
    - Weak teachers (baseline models) provide initial guidance
    - Strong students (RL-enhanced models) learn from multiple teachers
    - Agents can learn from each other's positive behaviors
    - Shared critic provides consistent value estimates
    """
    
    def __init__(
        self,
        agents: List[AgentConfig],
        shared_critic: CriticAgent,
        reward_calculator: RewardCalculator,
        device: torch.device,
        config: Dict = None
    ):
        self.agents = agents
        self.shared_critic = shared_critic
        self.reward_calculator = reward_calculator
        self.device = device
        
        # MACPO-inspired configuration
        self.config = config or {
            'ppo_epochs': 4,
            'ppo_clip_epsilon': 0.2,
            'ppo_entropy_coef': 0.01,
            'gae_lambda': 0.95,
            'value_loss_coef': 0.5,
            'max_grad_norm': 1.0,
            'mutual_learning_weight': 0.3,  # Weight for learning from other agents
            'self_confidence_weight': 0.7,  # Weight for self-generated responses
        }
        
        # Initialize agent models and optimizers
        self.agent_models = {}
        self.agent_optimizers = {}
        self._initialize_agents()
        
        # Training history
        self.training_history = {
            'agent_rewards': {agent.name: [] for agent in agents},
            'shared_critic_loss': [],
            'mutual_learning_gains': []
        }
    
    def _initialize_agents(self):
        """Initialize all agent models and optimizers."""
        for agent_config in self.agents:
            # Load agent model (this would be your ECR model)
            # For now, we'll create a placeholder
            agent_model = self._load_agent_model(agent_config)
            agent_model.to(self.device)
            
            # Initialize optimizer
            optimizer = optim.AdamW(
                agent_model.parameters(),
                lr=agent_config.learning_rate,
                weight_decay=agent_config.weight_decay
            )
            
            self.agent_models[agent_config.name] = agent_model
            self.agent_optimizers[agent_config.name] = optimizer
            
            logging.info(f"Initialized agent: {agent_config.name} ({agent_config.specialization})")
    
    def _load_agent_model(self, agent_config: AgentConfig):
        """Load agent model from path."""
        # This would load your actual ECR model
        # For now, return a placeholder
        return nn.Module()  # Placeholder
    
    def mutual_positive_behavior_augmentation(
        self, 
        context: str, 
        agent_responses: Dict[str, str]
    ) -> Dict[str, float]:
        """
        MACPO-inspired mutual positive behavior augmentation.
        
        Agents learn from each other's positive behaviors by:
        1. Evaluating each other's responses
        2. Identifying high-quality positive behaviors
        3. Using these as learning targets
        """
        mutual_learning_scores = {}
        
        for agent_name, response in agent_responses.items():
            # Calculate quality scores for this response
            quality_scores = self.reward_calculator.calculate_reward_breakdown(context, response)
            overall_score = sum(quality_scores.values()) / len(quality_scores)
            
            # Get critic's evaluation
            with torch.no_grad():
                critic_value = self.shared_critic.evaluate_response(context, response)
            
            # Combine reward calculator and critic scores
            combined_score = 0.7 * overall_score + 0.3 * critic_value
            
            mutual_learning_scores[agent_name] = combined_score
        
        return mutual_learning_scores
    
    def hard_negative_behavior_construction(
        self, 
        context: str, 
        agent_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """
        MACPO-inspired hard negative behavior construction.
        
        Generate familiar negative behaviors that agents should avoid.
        """
        negative_behaviors = {}
        
        for agent_name, response in agent_responses.items():
            # Create negative variants of responses
            negative_variants = self._create_negative_variants(response)
            
            # Select the most familiar (hardest) negative
            hardest_negative = self._select_hardest_negative(context, negative_variants)
            negative_behaviors[agent_name] = hardest_negative
        
        return negative_behaviors
    
    def _create_negative_variants(self, response: str) -> List[str]:
        """Create negative variants of a response."""
        # Simple negative variants (in practice, you'd use more sophisticated methods)
        variants = [
            response + " [insensitive]",
            response + " [offensive]",
            response + " [irrelevant]",
            response + " [repetitive]",
        ]
        return variants
    
    def _select_hardest_negative(self, context: str, variants: List[str]) -> str:
        """Select the hardest negative variant to avoid."""
        # For now, return the first variant
        # In practice, you'd use the critic to evaluate which is hardest
        return variants[0]
    
    def train_step(
        self, 
        batch: Dict, 
        agent_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Single training step for all agents.
        
        Implements MACPO-inspired training:
        1. Generate responses from all agents
        2. Evaluate responses using shared critic
        3. Apply mutual positive behavior augmentation
        4. Apply hard negative behavior construction
        5. Update all agents with PPO
        """
        if agent_names is None:
            agent_names = list(self.agent_models.keys())
        
        context = batch['context']
        target_response = batch.get('response', '')
        
        # Step 1: Generate responses from all agents
        agent_responses = {}
        for agent_name in agent_names:
            response = self._generate_response(agent_name, context)
            agent_responses[agent_name] = response
        
        # Step 2: Mutual positive behavior augmentation (MACPO)
        mutual_scores = self.mutual_positive_behavior_augmentation(context, agent_responses)
        
        # Step 3: Hard negative behavior construction (MACPO)
        negative_behaviors = self.hard_negative_behavior_construction(context, agent_responses)
        
        # Step 4: Calculate rewards for each agent
        agent_rewards = {}
        for agent_name in agent_names:
            response = agent_responses[agent_name]
            
            # Base reward from reward calculator
            base_reward = self.reward_calculator.calculate_reward(context, response)
            
            # Mutual learning reward (MACPO)
            mutual_reward = mutual_scores[agent_name]
            
            # Negative behavior penalty (MACPO)
            negative_penalty = self._calculate_negative_penalty(
                context, response, negative_behaviors[agent_name]
            )
            
            # Combined reward
            total_reward = (
                self.config['self_confidence_weight'] * base_reward +
                self.config['mutual_learning_weight'] * mutual_reward -
                negative_penalty
            )
            
            agent_rewards[agent_name] = total_reward
        
        # Step 5: Update all agents with PPO
        update_losses = {}
        for agent_name in agent_names:
            loss = self._update_agent_ppo(
                agent_name, context, agent_responses[agent_name], agent_rewards[agent_name]
            )
            update_losses[agent_name] = loss
        
        # Step 6: Update shared critic
        critic_loss = self._update_shared_critic(context, agent_responses, agent_rewards)
        
        # Update training history
        for agent_name, reward in agent_rewards.items():
            self.training_history['agent_rewards'][agent_name].append(reward)
        self.training_history['shared_critic_loss'].append(critic_loss)
        
        return {
            'agent_rewards': agent_rewards,
            'agent_losses': update_losses,
            'critic_loss': critic_loss,
            'mutual_scores': mutual_scores
        }
    
    def _generate_response(self, agent_name: str, context: str) -> str:
        """Generate response from a specific agent."""
        # This would use your actual ECR model to generate responses
        # For now, return a placeholder
        return f"Response from {agent_name}: {context[:50]}..."
    
    def _calculate_negative_penalty(
        self, 
        context: str, 
        response: str, 
        negative_behavior: str
    ) -> float:
        """Calculate penalty for negative behavior."""
        # Calculate similarity between response and negative behavior
        # Higher similarity = higher penalty
        similarity = self._calculate_similarity(response, negative_behavior)
        return similarity * 0.5  # Penalty coefficient
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_agent_ppo(
        self, 
        agent_name: str, 
        context: str, 
        response: str, 
        reward: float
    ) -> float:
        """Update a specific agent using PPO."""
        agent_model = self.agent_models[agent_name]
        optimizer = self.agent_optimizers[agent_name]
        
        # Get critic value for advantage calculation
        with torch.no_grad():
            critic_value = self.shared_critic.evaluate_response(context, response)
        
        # Calculate advantage
        advantage = reward - critic_value
        
        # PPO update (simplified)
        # In practice, you'd implement full PPO with policy ratio clipping
        loss = -advantage  # Simplified policy gradient loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_model.parameters(), self.config['max_grad_norm'])
        optimizer.step()
        
        return loss.item()
    
    def _update_shared_critic(
        self, 
        context: str, 
        agent_responses: Dict[str, str], 
        agent_rewards: Dict[str, float]
    ) -> float:
        """Update the shared critic."""
        # Calculate target values (actual rewards)
        target_values = torch.tensor(list(agent_rewards.values()), device=self.device)
        
        # Get critic predictions
        predicted_values = []
        for response in agent_responses.values():
            value = self.shared_critic.evaluate_response(context, response)
            predicted_values.append(value)
        
        predicted_values = torch.tensor(predicted_values, device=self.device)
        
        # Calculate critic loss
        critic_loss = nn.MSELoss()(predicted_values, target_values)
        
        # Update critic
        critic_optimizer = optim.AdamW(self.shared_critic.parameters(), lr=1e-5)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.config['max_grad_norm'])
        critic_optimizer.step()
        
        return critic_loss.item()
    
    def train(
        self, 
        dataloader: DataLoader, 
        num_epochs: int,
        save_path: str = None
    ):
        """Train all agents for multiple epochs."""
        logging.info(f"Starting multi-agent PPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_rewards = {agent.name: [] for agent in self.agents}
            epoch_losses = {agent.name: [] for agent in self.agents}
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                # Training step
                step_results = self.train_step(batch)
                
                # Update progress tracking
                for agent_name, reward in step_results['agent_rewards'].items():
                    epoch_rewards[agent_name].append(reward)
                
                for agent_name, loss in step_results['agent_losses'].items():
                    epoch_losses[agent_name].append(loss)
                
                # Update progress bar
                avg_rewards = {name: np.mean(rewards) for name, rewards in epoch_rewards.items()}
                progress_bar.set_postfix(avg_rewards)
            
            # Log epoch results
            for agent_name in self.agents:
                avg_reward = np.mean(epoch_rewards[agent_name.name])
                avg_loss = np.mean(epoch_losses[agent_name.name])
                logging.info(f"Epoch {epoch+1} - {agent_name.name}: "
                           f"Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Save models periodically
            if save_path and (epoch + 1) % 5 == 0:
                self.save_models(f"{save_path}/epoch_{epoch+1}")
        
        logging.info("Multi-agent PPO training completed")
    
    def save_models(self, save_path: str):
        """Save all agent models and shared critic."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save agent models
        for agent_name, model in self.agent_models.items():
            agent_path = os.path.join(save_path, f"{agent_name}_model.pth")
            torch.save(model.state_dict(), agent_path)
        
        # Save shared critic
        critic_path = os.path.join(save_path, "shared_critic.pth")
        torch.save(self.shared_critic.state_dict(), critic_path)
        
        # Save training history
        history_path = os.path.join(save_path, "training_history.json")
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logging.info(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str):
        """Load all agent models and shared critic."""
        import os
        
        # Load agent models
        for agent_name in self.agent_models.keys():
            agent_path = os.path.join(load_path, f"{agent_name}_model.pth")
            if os.path.exists(agent_path):
                self.agent_models[agent_name].load_state_dict(torch.load(agent_path))
        
        # Load shared critic
        critic_path = os.path.join(load_path, "shared_critic.pth")
        if os.path.exists(critic_path):
            self.shared_critic.load_state_dict(torch.load(critic_path))
        
        logging.info(f"Models loaded from {load_path}") 