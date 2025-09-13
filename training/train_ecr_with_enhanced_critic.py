import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import random
import itertools
import os
import logging
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src_emo.rl.enhanced_critic_with_trained_model_v2 import EnhancedCriticAgentWithTrainedModelV2
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECRDataset(Dataset):
    """Dataset for ECR training with dialogue context and user preferences."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        """Load dialogue data with user preferences and construct candidate sets for NDCG."""
        rng = random.Random(42)
        records: List[Dict] = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                context = self.extract_context(item)
                user_preference = self.extract_user_preference(item)
                if context and user_preference:
                    records.append({
                        'context': context,
                        'user_preference': user_preference,
                        'original_response': item.get('resp', '')
                    })
        # Build a global item pool from all user preferences
        all_items = list(sorted(set(itertools.chain.from_iterable(r['user_preference'] for r in records))))
        all_items_set = set(all_items)
        logger.info(f"Loaded {len(records)} raw samples; global item pool size: {len(all_items)}")

        # For each record, construct candidate_items = positives + sampled negatives
        data: List[Dict] = []
        num_negatives = 50  # negatives per sample; ensures NDCG is informative
        for r in records:
            positives = set(r['user_preference'])
            negatives_population = list(all_items_set - positives)
            if negatives_population:
                k = min(num_negatives, len(negatives_population))
                negatives = rng.sample(negatives_population, k=k)
            else:
                negatives = []
            candidate_items = list(positives) + negatives
            rng.shuffle(candidate_items)
            data.append({
                'context': r['context'],
                'user_preference': r['user_preference'],
                'original_response': r['original_response'],
                # policy will pick a top-k list from these candidates
                'candidate_items': candidate_items
            })

        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def extract_context(self, item: Dict) -> str:
        """Extract dialogue context from data item."""
        # Handle the actual data format from ReDial
        if 'context' in item and isinstance(item['context'], list):
            # Join context turns into a single string
            context = " ".join([turn for turn in item['context'] if turn.strip()])
            return context
        return ""
    
    def extract_user_preference(self, item: Dict) -> List[str]:
        """Extract user preferences from data item."""
        # Handle the actual data format from ReDial
        if 'rec' in item and isinstance(item['rec'], list):
            # Convert movie IDs to strings for NDCG calculation
            return [str(movie_id) for movie_id in item['rec'] if movie_id]
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function to handle variable-length data."""
    # Extract all fields
    contexts = [item['context'] for item in batch]
    user_preferences = [item['user_preference'] for item in batch]
    original_responses = [item['original_response'] for item in batch]
    candidate_items = [item['candidate_items'] for item in batch]
    
    return {
        'context': contexts,
        'user_preference': user_preferences,
        'original_response': original_responses,
        'candidate_items': candidate_items
    }


class ECRPolicy(nn.Module):
    """Policy network for ECR response generation."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"ECR Policy initialized with {model_name}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for response generation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate_response(self, context: str, max_length: int = 100) -> str:
        """Generate response given context."""
        # Get device
        device = next(self.parameters()).device
        
        # Tokenize input and move to device
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        inputs.input_ids = inputs.input_ids.to(device)
        inputs.attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def get_recommendations(self, context: str, response: str, candidate_items: List[str], top_k: int = 50) -> List[str]:
        """Select top-k recommendations from candidate pool.
        Currently returns the first top_k; replace with a trained ranker later.
        """
        return candidate_items[:top_k] if candidate_items else []


class PPOTrainer:
    """PPO trainer for ECR with enhanced critic."""
    
    def __init__(self, policy: ECRPolicy, critic: EnhancedCriticAgentWithTrainedModelV2, 
                 lr: float = 1e-5, clip_epsilon: float = 0.2, 
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.policy = policy
        self.critic = critic
        self.optimizer = optim.AdamW(policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        logger.info("PPO Trainer initialized")
    
    def compute_ppo_loss(self, states, actions, old_log_probs, advantages, returns):
        """Compute PPO loss."""
        # Forward pass
        outputs = self.policy(states, labels=actions)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get action log probabilities
        action_log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute ratio
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss (simplified)
        value_loss = 0.0  # We're using critic for value estimation
        
        # Compute entropy loss
        entropy_loss = -outputs.logits.softmax(dim=-1).log_softmax(dim=-1).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        # Handle batch format - batch contains lists of items
        contexts = batch['context']
        user_preferences = batch['user_preference']
        candidate_items_list = batch.get('candidate_items', [])
        
        # For now, process only the first item in the batch
        # In a full implementation, you'd process all items
        context = contexts[0] if isinstance(contexts, list) else contexts
        user_preference = user_preferences[0] if isinstance(user_preferences, list) else user_preferences
        candidate_items = candidate_items_list[0] if isinstance(candidate_items_list, list) else candidate_items_list
        
        # Generate response using current policy
        response = self.policy.generate_response(context)
        
        # Get recommendations (top-K per CRSDP; default K=50)
        recommended_items = self.policy.get_recommendations(context, response, candidate_items, top_k=50)
        
        # Ensure all tensors are on the same device
        device = next(self.policy.parameters()).device
        
        # Get comprehensive reward from critic
        total_reward, detailed_scores = self.critic.get_comprehensive_reward(
            context=context,
            response=response,
            recommended_items=recommended_items,
            user_preference=user_preference,
            k=50
        )
        
        # Convert reward to tensor and move to device
        reward = torch.tensor(total_reward, dtype=torch.float32, device=device)
        
        # REINFORCE-style objective: maximize log-prob of generated response weighted by reward
        # Tokenize context and response separately to mask context in labels
        ctx_enc = self.policy.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        resp_enc = self.policy.tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=False
        )

        input_ids = torch.cat([ctx_enc.input_ids, resp_enc.input_ids], dim=1).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()
        ctx_len = ctx_enc.input_ids.size(1)
        labels[:, :ctx_len] = -100  # mask context tokens; compute loss on response tokens only

        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        nll_loss = outputs.loss  # average negative log-likelihood over response tokens
        loss = reward * nll_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'reward': reward.item(),
            'detailed_scores': detailed_scores
        }


def train_ecr_with_enhanced_critic(args):
    """Main training function."""
    logger.info("Starting ECR training with enhanced critic...")
    
    # Initialize models
    policy = ECRPolicy(model_name=args.policy_model)
    # Use the newly trained v2 critic with the saved checkpoint
    critic = EnhancedCriticAgentWithTrainedModelV2(trained_model_path="critic_roberta_best_v2.pth")
    
    # Move to device
    device = torch.device(args.device)
    policy.to(device)
    critic.to(device)
    
    # Initialize trainer
    trainer = PPOTrainer(policy, critic, lr=args.learning_rate)
    
    # Load dataset
    dataset = ECRDataset(args.data_path, policy.tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_rewards = []
        epoch_scores = {
            'empathy': [], 'persuasiveness': [], 'logic': [], 
            'informativeness': [], 'lifelikeness': [], 'recommendation_accuracy': []
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step - batch is now a single sample due to batch_size=1
            try:
                step_results = trainer.train_step(batch)
            except Exception as e:
                logger.warning(f"Error processing sample {batch_idx}: {e}")
                continue
            
            # Collect metrics
            epoch_losses.append(step_results['loss'])
            epoch_rewards.append(step_results['reward'])
            
            # Collect detailed scores
            for dimension, score in step_results['detailed_scores'].items():
                if dimension in epoch_scores:
                    epoch_scores[dimension].append(score)
            
            # Update progress bar
            avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) > 0 else 0
            avg_reward = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) > 0 else 0
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'reward': f'{avg_reward:.4f}'
            })
            
            # Save checkpoint periodically
            if (batch_idx + 1) % args.save_steps == 0:
                save_checkpoint(policy, critic, args.output_dir, epoch, batch_idx)
        
        # Log epoch results
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        avg_scores = {dim: np.mean(scores) for dim, scores in epoch_scores.items()}
        
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Average Reward: {avg_reward:.4f}")
        logger.info("  Average Scores:")
        for dimension, score in avg_scores.items():
            logger.info(f"    {dimension}: {score:.3f}")
        
        # Save epoch checkpoint
        save_checkpoint(policy, critic, args.output_dir, epoch, -1)
    
    logger.info("Training completed!")


def save_checkpoint(policy: ECRPolicy, critic: EnhancedCriticAgentWithTrainedModelV2, 
                   output_dir: str, epoch: int, step: int):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'policy_tokenizer': policy.tokenizer,
        'critic_tokenizer': critic.tokenizer,
    }
    
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt" if step >= 0 else f"checkpoint_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(output_dir, filename)
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ECR with Enhanced Critic")
    
    # Model arguments
    parser.add_argument("--policy_model", type=str, default="microsoft/DialoGPT-small",
                       help="Policy model name")
    parser.add_argument("--critic_model", type=str, default="roberta-base",
                       help="Critic model name")
    
    # Training arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./ecr_enhanced_critic_output",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_ecr_with_enhanced_critic(args)


if __name__ == "__main__":
    main() 