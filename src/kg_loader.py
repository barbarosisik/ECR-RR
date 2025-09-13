import json
import os
from typing import Dict, List, Tuple, Optional


class KnowledgeAssetsLoader:
    """
    Lightweight loader for ECR KG assets (movie -> entity, entity neighbors/triples).
    Falls back gracefully if files are missing.
    Caches small normalized maps under data_root/cache/.
    """

    def __init__(self, data_root: str = "src_emo/data/redial_gen"):
        self.data_root = data_root
        self.cache_root = os.path.join(self.data_root, "cache")
        os.makedirs(self.cache_root, exist_ok=True)

        # Candidate filenames (we'll probe multiple common names)
        self.movie_to_entity_filenames = [
            "movie_id2entity.json",
            "movie2entity.json",
            "movie_entity_map.json",
        ]
        self.entity_neighbors_filenames = [
            "entity2neighbors.json",
            "entity_neighbors.json",
        ]
        self.entity_triples_filenames = [
            "entity_triples.json",
            "triples.json",
        ]

        # Loaded maps
        self._movie_to_entity: Dict[str, str] = {}
        self._entity_neighbors: Dict[str, List[str]] = {}
        self._entity_triples: Dict[str, List[Tuple[str, str]]] = {}

        self._load_all()

    def _first_existing(self, filenames: List[str]) -> Optional[str]:
        for fname in filenames:
            path = os.path.join(self.data_root, fname)
            if os.path.isfile(path):
                return path
        return None

    def _load_json(self, path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_all(self) -> None:
        # movie -> entity
        movie_path = self._first_existing(self.movie_to_entity_filenames)
        if movie_path:
            data = self._load_json(movie_path)
            if isinstance(data, dict):
                # Normalize keys to str
                self._movie_to_entity = {str(k): str(v) for k, v in data.items()}

        # entity -> neighbors
        neigh_path = self._first_existing(self.entity_neighbors_filenames)
        if neigh_path:
            data = self._load_json(neigh_path)
            if isinstance(data, dict):
                self._entity_neighbors = {str(k): [str(e) for e in v] for k, v in data.items()}

        # entity -> triples (relation, neighbor)
        triples_path = self._first_existing(self.entity_triples_filenames)
        if triples_path:
            data = self._load_json(triples_path)
            if isinstance(data, dict):
                norm = {}
                for ent, triples in data.items():
                    cleaned = []
                    if isinstance(triples, list):
                        for t in triples:
                            if isinstance(t, (list, tuple)) and len(t) >= 2:
                                rel, tail = t[0], t[1]
                                cleaned.append((str(rel), str(tail)))
                    norm[str(ent)] = cleaned
                self._entity_triples = norm

    # Public API
    def get_movie_entity(self, movie_id: str) -> Optional[str]:
        return self._movie_to_entity.get(str(movie_id))

    def get_entity_neighbors(self, entity: str) -> List[str]:
        return self._entity_neighbors.get(str(entity), [])

    def get_entity_triples(self, entity: str) -> List[Tuple[str, str]]:
        return self._entity_triples.get(str(entity), [])

    def has_assets(self) -> bool:
        return bool(self._movie_to_entity)

