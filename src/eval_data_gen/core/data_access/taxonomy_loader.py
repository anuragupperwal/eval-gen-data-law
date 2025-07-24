# src/eval_data_gen/core/data_access/taxonomy_loader.py
import json, yaml
from pathlib import Path

def _flatten(domain, cat_name, item):
    leaf_id = f"{domain}.{cat_name}.{item['topic'].split(' – ')[0].replace(' ', '')}"
    return {
        "id": leaf_id,
        "label": item["topic"],
        "difficulty": item["difficulty"],
    }

def load_taxonomy(path):
    p = Path(path)                  
    data = yaml.safe_load(p.read_text())
    domain = data["domain"]
    leaves = []
    for cate, items in data["categories"].items():
        for item in items:
            leaves.append(_flatten(domain, cate, item))
    return leaves