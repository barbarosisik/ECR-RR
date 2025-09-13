import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel


class TrainedCriticRobertaMultiHeadV2(nn.Module):
    def __init__(self, model_name: str = "roberta-base", num_outputs: int = 5, dropout: float = 0.2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size
        self.pooler = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.pooler(pooled)
        return self.head(pooled)


class EnhancedCriticAgentWithTrainedModelV2(nn.Module):
    def __init__(self, trained_model_path: str = "critic_roberta_best_v2.pth", num_dimensions: int = 6):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.trained_critic = TrainedCriticRobertaMultiHeadV2()
        if os.path.exists(trained_model_path):
            self.trained_critic.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
        # Move once to the active device and set eval mode to avoid repeated .to() calls per forward
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_critic = self.trained_critic.to(self._device)
        self.trained_critic.eval()
        self.recommendation_head = nn.Linear(768, 1)
        self.default_weights = {
            'empathy': 0.25,
            'persuasiveness': 0.15,
            'logic': 0.15,
            'informativeness': 0.10,
            'lifelikeness': 0.10,
            'recommendation_accuracy': 0.25,
        }
        self.dimension_mapping = {
            0: 'empathy',
            1: 'persuasiveness',
            2: 'logic',
            3: 'informativeness',
            4: 'lifelikeness',
        }

    def forward(self, context: str, response: str) -> Dict[str, torch.Tensor]:
        device = self._device
        input_text = f"<context>{context}</context><response>{response}</response>"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            scores = self.trained_critic(inputs['input_ids'], inputs['attention_mask']).squeeze(0)
        return {name: scores[idx].unsqueeze(0) for idx, name in self.dimension_mapping.items()}

    @staticmethod
    def calculate_ndcg(recommended_items: List[str], user_preference: List[str], k: int = 10) -> float:
        """Compute NDCG@k given a ranked list and a set of relevant items.
        Items in user_preference are treated as relevant (gain=1), others 0.
        """
        if not recommended_items or not user_preference:
            return 0.0
        rel = [1.0 if item in user_preference else 0.0 for item in recommended_items[:k]]
        dcg = 0.0
        for i, r in enumerate(rel):
            if r > 0:
                dcg += r / np.log2(i + 2)
        ideal_positives = min(len(user_preference), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_positives))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def get_comprehensive_reward(
        self,
        context: str,
        response: str,
        recommended_items: List[str],
        user_preference: List[str],
        weights: Optional[Dict[str, float]] = None,
        k: int = 10,
    ) -> Tuple[float, Dict[str, float]]:
        if weights is None:
            weights = self.default_weights
        subjective = self.forward(context, response)
        rec = self.calculate_ndcg(recommended_items, user_preference, k)
        total = 0.0
        comps: Dict[str, float] = {}
        for dim, score in subjective.items():
            if dim in weights:
                comps[dim] = score.item()
                total += score.item() * weights[dim]
        if 'recommendation_accuracy' in weights:
            comps['recommendation_accuracy'] = rec
            total += rec * weights['recommendation_accuracy']
        return total, comps


