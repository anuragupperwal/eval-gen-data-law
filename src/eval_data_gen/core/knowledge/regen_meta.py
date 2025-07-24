from pathlib import Path
from typing import List
import json, gzip
from datasets import load_dataset
from eval_data_gen.core.utils.text_chunk import sent_merge   # or your chunk() func

DATASET_NAME = "ninadn/indian-legal"
TEXT_FIELD   = "Text"
CHUNK_SIZE   = 128
OUT_META     = Path("tmp/meta.jsonl.gz")

def main():
    ds = load_dataset(DATASET_NAME, split="train", streaming=False)

    with gzip.open(OUT_META, "wt") as gz:
        chunk_id = 0
        for doc_id, row in enumerate(ds):
            for part in sent_merge(row[TEXT_FIELD], target_tokens=CHUNK_SIZE):
                if len(part.split()) < 20:
                    continue
                rec = {
                    "chunk_id": chunk_id,
                    "doc_id":   doc_id,
                    "text":     part          # no truncation
                }
                gz.write(json.dumps(rec) + "\n")
                chunk_id += 1
    print(f"[regen_meta] Wrote {chunk_id:,} records → {OUT_META}")

if __name__ == "__main__":
    main()