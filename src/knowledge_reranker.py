from typing import Dict, List, Optional, Set


class KnowledgeReranker:
    """
    Lightweight entity-aware reranker with optional emotion feedback weights
    and recency bias. Designed to be a drop-in replacement for naive top-k.
    """

    def __init__(self, feedback_weights: Optional[Dict[str, float]] = None, recency_weight: float = 0.1):
        # feedback label -> scalar
        self.feedback_weights = feedback_weights or {
            "like": 2.0,
            "not_say": 1.0,
            "dislike": 0.5,
        }
        self.recency_weight = recency_weight

    def score_items(
        self,
        candidate_items: List[str],
        dialogue_entities: Optional[List[str]] = None,
        item_to_entities: Optional[Dict[str, List[str]]] = None,
        feedback_by_item: Optional[Dict[str, str]] = None,
        recent_user_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        ent_set: Set[str] = set(dialogue_entities or [])
        recent_set: Set[str] = set(recent_user_entities or [])
        scores: Dict[str, float] = {}

        for it in candidate_items:
            s = 0.0
            ents = (item_to_entities or {}).get(str(it), [])
            # entity overlap
            overlap = len(ent_set.intersection(ents))
            s += float(overlap)
            # recency bias: overlap with last-turn entities counts extra
            rec_overlap = len(recent_set.intersection(ents))
            s += self.recency_weight * float(rec_overlap)
            # feedback weight
            if feedback_by_item and str(it) in feedback_by_item:
                label = feedback_by_item[str(it)]
                s *= self.feedback_weights.get(label, 1.0)
            scores[str(it)] = s
        return scores

    def rerank(
        self,
        candidate_items: List[str],
        dialogue_entities: Optional[List[str]] = None,
        item_to_entities: Optional[Dict[str, List[str]]] = None,
        feedback_by_item: Optional[Dict[str, str]] = None,
        recent_user_entities: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[str]:
        scores = self.score_items(
            candidate_items,
            dialogue_entities=dialogue_entities,
            item_to_entities=item_to_entities,
            feedback_by_item=feedback_by_item,
            recent_user_entities=recent_user_entities,
        )
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [it for it, _ in ordered[:top_k]]

