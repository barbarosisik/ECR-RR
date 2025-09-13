from typing import Dict, List
from .kg_loader import KnowledgeAssetsLoader


class KnowledgeRetriever:
    """Select a tiny, stable knowledge block for prompts.

    For a given item id, returns up to 2 triples and 4 entities related to the item
    via the movie->entity map and entity neighbors/triples.
    """

    def __init__(self, data_root: str = "src_emo/data/redial_gen"):
        self.loader = KnowledgeAssetsLoader(data_root=data_root)

    def build_block_for_items(self, item_ids: List[str], max_triples: int = 2, max_entities: int = 4) -> str:
        triples_out: List[str] = []
        entities_out: List[str] = []

        for item_id in item_ids:
            ent = self.loader.get_movie_entity(str(item_id))
            if not ent:
                continue

            # Triples for entity
            for rel, tail in self.loader.get_entity_triples(ent):
                if len(triples_out) >= max_triples:
                    break
                triples_out.append(f"{ent} --{rel}--> {tail}")

            # Neighbors
            for nb in self.loader.get_entity_neighbors(ent):
                if len(entities_out) >= max_entities:
                    break
                entities_out.append(nb)

            if len(triples_out) >= max_triples and len(entities_out) >= max_entities:
                break

        triples_str = ", ".join(triples_out) if triples_out else "None"
        entities_str = ", ".join(entities_out) if entities_out else "None"

        return f"[KNOW] Triples: {triples_str}; Entities: {entities_str} [/KNOW]"

