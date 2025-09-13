import math
from typing import Dict, List, Optional, Set


class KnowledgeReranker:
    """
    Lightweight knowledge-/emotion-aware reranker for candidate items.

    - Prefers items mentioned as entities in the dialogue (entity overlap)
    - Applies feedback-aware item reweighting (like/dislike/not say)
    - Optional prior score integration

    This is a minimal, dependency-free reranker designed to be swapped out with
    a full KG model later without changing the calling code.
    """

    def __init__(
        self,
        feedback_weights: Optional[Dict[str, float]] = None,
        entity_overlap_weight: float = 1.0,
        prior_score_weight: float = 0.0,
    ) -> None:
        # ECR (RecSys'24) suggests like=2.0, dislike=1.0, not say=0.5
        self.feedback_weights: Dict[str, float] = feedback_weights or {
            "like": 2.0,
            "dislike": 1.0,
            "not_say": 0.5,
        }
        self.entity_overlap_weight = float(entity_overlap_weight)
        self.prior_score_weight = float(prior_score_weight)

    def _feedback_scalar(self, item_id: str, item_feedback: Optional[Dict[str, str]]) -> float:
        if not item_feedback:
            return 1.0
        fb = item_feedback.get(item_id)
        if not fb:
            return 1.0
        # Normalize key variants
        key = fb.lower().replace(" ", "_")
        return float(self.feedback_weights.get(key, 1.0))

    def _overlap_score(self, item_id: str, dialogue_entities: Optional[Set[str]]) -> float:
        if not dialogue_entities:
            return 0.0
        return self.entity_overlap_weight if item_id in dialogue_entities else 0.0

    def rerank(
        self,
        candidate_items: List[str],
        dialogue_entities: Optional[List[str]] = None,
        item_feedback: Optional[Dict[str, str]] = None,
        prior_scores: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> List[str]:
        """
        Rank candidates by entity overlap and feedback-aware weighting.

        Args:
            candidate_items: list of movie IDs (strings)
            dialogue_entities: list of movie/entity IDs mentioned in dialogue
            item_feedback: mapping item_id -> {like|dislike|not_say}
            prior_scores: optional prior scores for items
            top_k: return top-K
        Returns:
            Ranked list (top_k) of item IDs
        """
        entities_set: Set[str] = set(dialogue_entities or [])
        scores: List[tuple[str, float]] = []

        for item in candidate_items:
            s = 0.0
            # entity overlap
            s += self._overlap_score(item, entities_set)
            # prior
            if prior_scores and item in prior_scores:
                s += self.prior_score_weight * float(prior_scores[item])
            # feedback scalar
            s *= self._feedback_scalar(item, item_feedback)
            scores.append((item, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scores[: max(1, top_k)]]

