import json
from typing import List, Dict

from src_emo.rl.enhanced_critic_with_trained_model import EnhancedCriticAgentWithTrainedModel


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def ndcg_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    if not recommended or not ground_truth:
        return 0.0
    rel = [1.0 if it in ground_truth else 0.0 for it in recommended[:k]]
    dcg = sum(rel[i] / __import__('math').log2(i + 2) for i in range(len(rel)))
    ideal = sorted(rel, reverse=True)
    idcg = sum(ideal[i] / __import__('math').log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0


def eval_subjective(critic: EnhancedCriticAgentWithTrainedModel, samples: List[Dict], max_n: int = 200) -> Dict[str, float]:
    scores_acc = {"empathy": 0.0, "persuasiveness": 0.0, "logic": 0.0, "informativeness": 0.0, "lifelikeness": 0.0}
    n = 0
    for item in samples[:max_n]:
        context = " ".join(item.get("context", []))
        resp = item.get("resp", "")
        if not context or not resp:
            continue
        _, detailed = critic.get_comprehensive_reward(context, resp, [], [])
        for k in scores_acc.keys():
            if k in detailed:
                scores_acc[k] += detailed[k]
        n += 1
    if n == 0:
        return {k: 0.0 for k in scores_acc.keys()}
    return {k: v / n for k, v in scores_acc.items()}


def eval_recommendation(samples: List[Dict], k: int = 10, max_n: int = 1000) -> float:
    # expects fields: rec (recommended candidates), entity (ground_truth relevant) or similar
    total = 0.0
    n = 0
    for item in samples[:max_n]:
        rec = [str(x) for x in item.get("rec", [])]
        gt = [str(x) for x in item.get("entity", [])]
        if not rec or not gt:
            continue
        total += ndcg_at_k(rec, gt, k=k)
        n += 1
    return total / n if n > 0 else 0.0


def run_all(data_path: str, critic_path: str) -> None:
    samples = load_jsonl(data_path)
    critic = EnhancedCriticAgentWithTrainedModel(trained_model_path=critic_path)
    subj = eval_subjective(critic, samples)
    ndcg10 = eval_recommendation(samples, k=10)
    ndcg50 = eval_recommendation(samples, k=50)
    print("Subjective (critic):", subj)
    print("NDCG@10:", round(ndcg10, 4))
    print("NDCG@50:", round(ndcg50, 4))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--critic_path", default="critic_roberta_best.pth")
    args = ap.parse_args()
    run_all(args.data_path, args.critic_path)

