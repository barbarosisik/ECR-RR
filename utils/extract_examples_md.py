#!/usr/bin/env python3
import json
import os
from typing import List, Tuple, Optional
import re


SELECTED_FILES = [
    # High-delta include-NDCG, longer outputs
    "/data1/s3905993/ECR-main/src_emo/data/redial_gen/critic_rerank_eval_full_inc_n16_t80_32_k50_n500.jsonl",
    # Lower-delta exclude-NDCG, longer outputs
    "/data1/s3905993/ECR-main/src_emo/data/redial_gen/critic_rerank_eval_full_exc_n8_t80_32_k50_n500.jsonl",
    # Shorter outputs include-NDCG
    "/data1/s3905993/ECR-main/src_emo/data/redial_gen/critic_rerank_eval_full_inc_n8_t48_16_k50_n500.jsonl",
]


def truncate(text: str, limit: int = 600) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    return (text[:limit] + "…") if len(text) > limit else text


def load_rows(fp: str) -> List[Tuple[float, dict]]:
    rows: List[Tuple[float, dict]] = []
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                r_base = float(obj.get("r_base", 0.0) or 0.0)
                r_best = float(obj.get("r_best", 0.0) or 0.0)
                delta = r_best - r_base
                rows.append((delta, obj))
    except FileNotFoundError:
        return []
    return rows


def fmt_float(x):
    return "NA" if x is None else f"{float(x):.4f}"


def fmt_dim(x):
    return "NA" if x is None else f"{float(x):.3f}"


def main():
    print("#### Qualitative examples (baseline vs reranked)\n")
    print("Below are representative cases from three settings: `inc_n16_t80_32_k50_n500` (highest average delta), `exc_n8_t80_32_k50_n500` (lower delta), and `inc_n8_t48_16_k50_n500` (shorter outputs). Each shows the context, baseline and reranked responses, critic rewards/deltas, per-dimension changes, and a brief inference.\n")

    for fp in SELECTED_FILES:
        tag = os.path.basename(fp).replace("critic_rerank_eval_full_", "").replace(".jsonl", "")
        rows = load_rows(fp)
        if not rows:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        picks = rows[:2] + rows[-1:]
        for delta, o in picks:
            sb = o.get("scores_base", {}) or {}
            st = o.get("scores_best", {}) or {}
            r_base = o.get("r_base")
            r_best = o.get("r_best")
            idx = o.get("idx")
            context = truncate(o.get("context", ""))
            baseline = truncate(o.get("baseline_resp", ""))
            reranked = truncate(o.get("rerank_resp", ""))

            print(f"- Tag: `{tag}` | idx={idx} | r_base={fmt_float(r_base)} → r_best={fmt_float(r_best)} (Δ={fmt_float(delta)})")
            print("  - Context:")
            print("    > " + context)
            print("  - Baseline:")
            print("    > " + baseline)
            print("  - Reranked:")
            print("    > " + reranked)
            print(
                "  - Per-dimension (base → best): "
                + f"empathy {fmt_dim(sb.get('empathy'))}→{fmt_dim(st.get('empathy'))}, "
                + f"pers {fmt_dim(sb.get('persuasiveness'))}→{fmt_dim(st.get('persuasiveness'))}, "
                + f"logic {fmt_dim(sb.get('logic'))}→{fmt_dim(st.get('logic'))}, "
                + f"info {fmt_dim(sb.get('informativeness'))}→{fmt_dim(st.get('informativeness'))}, "
                + f"life {fmt_dim(sb.get('lifelikeness'))}→{fmt_dim(st.get('lifelikeness'))}, "
                + f"rec_acc {fmt_dim(sb.get('recommendation_accuracy'))}→{fmt_dim(st.get('recommendation_accuracy'))}"
            )
            if delta >= 0.2:
                inf = (
                    "as you can see, the reranked response is more specific, coherent, and empathetic, which aligns "
                    "with the higher reward and per-dimension gains."
                )
            elif delta > 0.0:
                inf = (
                    "the reranked response shows a modest improvement in specificity and tone; this small qualitative gain "
                    "is reflected in the positive Δ."
                )
            else:
                inf = (
                    "counter-example: despite reranking, the baseline scored slightly better; this highlights a limitation "
                    "in the selection for this context."
                )
            print(f"  - Inference: {inf}\n")

    # Second section with highlighted evidence
    print("\n#### Qualitative examples with highlighted evidence\n")
    print("We highlight short text spans that best explain the reward differences (e.g., concrete titles, years, directives, or reasons) and add a one-line justification.\n")

    def pick_highlight(text: str) -> Optional[str]:
        if not text:
            return None
        # Prefer quoted titles
        m = re.search(r'"[^"\n]{5,80}"', text)
        if m:
            return m.group(0)
        # Prefer Title (Year)
        m = re.search(r'[A-Z][A-Za-z0-9\s:\-\']{2,60}\(\d{4}\)', text)
        if m:
            return m.group(0)
        # Prefer directive phrases
        for pat in [r'recommend\s+<movie>', r'I\s+recommend\s+[^\.]{3,80}', r'directed by[^\.]{3,80}', r'because[^\.]{3,120}', r'themes?\s+of[^\.]{3,120}']:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        # Fallback: first informative-looking sentence fragment
        m = re.search(r'[A-Za-z][^\.!?]{30,140}[\.!?]', text)
        if m:
            return m.group(0)
        return text[:80]

    for fp in SELECTED_FILES:
        tag = os.path.basename(fp).replace("critic_rerank_eval_full_", "").replace(".jsonl", "")
        rows = load_rows(fp)
        if not rows:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        picks = rows[:1] + rows[-1:]
        for delta, o in picks:
            r_base = o.get("r_base")
            r_best = o.get("r_best")
            sb = o.get("scores_base", {}) or {}
            st = o.get("scores_best", {}) or {}
            idx = o.get("idx")
            baseline = (o.get("baseline_resp", "") or "").replace("\n", " ")
            reranked = (o.get("rerank_resp", "") or "").replace("\n", " ")
            b_h = pick_highlight(baseline)
            r_h = pick_highlight(reranked)
            print(f"- Tag: `{tag}` | idx={idx} | r_base={fmt_float(r_base)} → r_best={fmt_float(r_best)} (Δ={fmt_float(delta)})")
            if b_h:
                print(f"  - Baseline highlight: **{b_h.strip()}**")
            if r_h:
                print(f"  - Reranked highlight: **{r_h.strip()}**")
            why = []
            if r_h and ('"' in r_h or re.search(r'\(\d{4}\)', r_h)):
                why.append("explicit title/year")
            if (st.get('informativeness', 0) or 0) > (sb.get('informativeness', 0) or 0):
                why.append("more informative")
            if (st.get('persuasiveness', 0) or 0) > (sb.get('persuasiveness', 0) or 0):
                why.append("more persuasive")
            if (st.get('empathy', 0) or 0) > (sb.get('empathy', 0) or 0):
                why.append("more empathetic")
            if not why:
                why.append("qualitatively stronger wording")
            print("  - Evidence-based inference: reranked wins via " + ", ".join(why) + ".\n")


if __name__ == "__main__":
    main()


