import faiss, numpy as np, gzip, json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict

class FaissRetriever:
    def __init__(
        self,
        index_path: str = "tmp/faiss.index",
        meta_path: str  = "tmp/meta.jsonl.gz",
        emb_model: str  = "sentence-transformers/all-MiniLM-L6-v2", #"snowflake/snowflake-arctic-embed-l",
        
    ):
        self.index = faiss.read_index(index_path)
        self.encoder = SentenceTransformer(emb_model)
        self.meta   = self._load_meta(meta_path)

    @staticmethod
    def _load_meta(path) -> List[Dict]:
        meta = []
        with gzip.open(path, "rt") as gz:
            for line in gz:
                meta.append(json.loads(line))
        return meta

    def query(self, text: str, k: int = 3):
        q_emb = self.encoder.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        distance, indices = self.index.search(q_emb[None, :].astype("float32"), k)
        results = []
        for score, idx in zip(distance[0], indices[0]):
            if idx == -1:
                continue
            meta_row = self.meta[int(idx)]
            meta_row["score"] = float(score)
            results.append(meta_row)   # contains chunk_id, doc_id, text, score
        return results


# pip install -e .
# # Build index only if missing
# python -m eval_data_gen.core.knowledge.index_builder

# # Run FAISS retrieval demo
# eval-data-gen retrieve-faiss
