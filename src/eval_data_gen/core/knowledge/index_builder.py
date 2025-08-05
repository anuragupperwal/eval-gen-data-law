import numpy as np
import faiss, json, gzip, os, sys

from pathlib import Path
from typing import List
from datasets import load_dataset
from langchain_ollama.embeddings import OllamaEmbeddings
from tqdm import tqdm


DATASET_NAME = "ninadn/indian-legal"
SPLIT = "train"
TEXT_FIELD = "Text"

OLLAMA_EMB_MODEL = "nomic-embed-text"
BATCH_SIZE = 16

OUT_DIR    = Path("tmp")
OUT_DIR.mkdir(exist_ok=True)
VEC_FILE   = OUT_DIR / "chunks.npy"
META_FILE  = OUT_DIR / "meta.jsonl.gz"
INDEX_FILE = OUT_DIR / "faiss.index"





def build_index():
    if INDEX_FILE.exists():
        print("FAISS index already exists, nothing to do.")
        return

    print(f"Loading corpus '{DATASET_NAME}' and embedding model '{OLLAMA_EMB_MODEL}'...")
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=False)
    # ds = ds.select(range(100))

    embeddings = OllamaEmbeddings(model=OLLAMA_EMB_MODEL)

    vecs, metas = [], []
    for doc_id, row in tqdm(enumerate(ds), total=len(ds)):
        document_text = row[TEXT_FIELD]

        # Basic validation: ensure there is text to process.
        if not document_text or not isinstance(document_text, str) or len(document_text.split()) < 20:
            continue
        
        vec = embeddings.embed_documents(document_text)
        metas.append({"chunk_id": len(metas), "doc_id": doc_id, "text": document_text})
        vecs.append(vec)



    print("Embedding complete.")
    vecs = np.vstack(vecs)
    np.save(VEC_FILE, vecs)
    print("Saved", vecs.shape, ":", VEC_FILE)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_FILE))
    print("FAISS index:", INDEX_FILE)

    with gzip.open(META_FILE, "wt") as gz:
        for m in metas:
            gz.write(json.dumps(m) + "\n")
    print("Meta: ", META_FILE)

if __name__ == "__main__":
    build_index()



# python - <<'PY'
# from datasets import load_dataset
# ds = load_dataset("ninadn/indian-legal", split="train", streaming=False)
# print("Columns:", ds.column_names)
# PY



