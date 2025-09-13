import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Any


class CriticSupervisedDatasetV2(Dataset):
    """
    Supervised dataset for training the RoBERTa critic on 5 subjective dimensions.

    Labels (scaled to 0-9):
      - empathy            <- llama2_scores.empathy_score
      - persuasiveness     <- llama2_scores.recommendation_score (proxy)
      - logic              <- llama2_scores.overall_score (proxy for coherence/logic)
      - informativeness    <- llama2_scores.informativeness_score
      - lifelikeness       <- llama2_scores.engagement_score (proxy)

    Notes:
    - We purposefully exclude NDCG from supervised targets. NDCG is computed at evaluation time
      from recommended items and ground-truth entities and should not overwrite any subjective head.
    - Input text concatenates context and response with lightweight tags to mirror inference usage.
    """

    REQUIRED_KEYS = [
        "empathy_score",
        "informativeness_score",
        "recommendation_score",
        "engagement_score",
        "overall_score",
    ]

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                # Basic fields
                context_list = data.get("context", []) or []
                response = data.get("resp", "")

                # Build input text
                context_text = " ".join(context_list) if isinstance(context_list, list) else str(context_list)
                input_text = f"<context>{context_text}</context><response>{response}</response>"

                # Accept multiple scorer field names for merged datasets
                scores = (
                    data.get("llama2_scores")
                    or data.get("mistral7b_scores")
                    or data.get("scores")
                    or {}
                )
                if not all(k in scores for k in self.REQUIRED_KEYS):
                    # Skip samples without full set of scores
                    continue

                # Map to five subjective dimensions, scaled to 0-9
                target = [
                    float(scores["empathy_score"]) * 9.0,            # empathy
                    float(scores["recommendation_score"]) * 9.0,     # persuasiveness (proxy)
                    float(scores["overall_score"]) * 9.0,            # logic (proxy)
                    float(scores["informativeness_score"]) * 9.0,    # informativeness
                    float(scores["engagement_score"]) * 9.0,         # lifelikeness (proxy)
                ]

                self.samples.append({
                    "text": input_text,
                    "labels": target,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
        }


