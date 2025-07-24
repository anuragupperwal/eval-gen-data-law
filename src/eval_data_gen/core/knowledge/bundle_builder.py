import json, os
from pathlib import Path
from typing import List, Dict

from eval_data_gen.core.knowledge.faiss_retriever import FaissRetriever

class BundleBuilder:

    def __init__(self, out_dir: str = "tmp/bundles", k: int = 4):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.retr = FaissRetriever()

    def build(self, leaf: Dict):
        query_terms = [leaf["label"]] + leaf.get("synonyms", [])
        retrieved_docs   = self.retr.query(leaf["label"], k=self.k)
        passages = retrieved_docs

        bundle = {
            "taxonomy_id": leaf["id"],
            "query_terms": query_terms,
            "passages":    passages
        }
        out_path = self.out_dir / f"{leaf['id']}.json"
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
        print(f"{leaf['id']} → {out_path}")

        return bundle

