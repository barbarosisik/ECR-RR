from transformers import PreTrainedTokenizer


def ensure_knowledge_tokens(tokenizer: PreTrainedTokenizer) -> None:
    """Add special knowledge tokens once if missing."""
    added = False
    specials = {"additional_special_tokens": ["[KNOW]", "[/KNOW]"]}
    current = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    for tok in specials["additional_special_tokens"]:
        if tok not in current:
            added = True
            break
    if added:
        tokenizer.add_special_tokens(specials)


def prepend_knowledge_block(context: str, knowledge_block: str) -> str:
    if not knowledge_block:
        return context
    return f"{knowledge_block}\n{context}"

