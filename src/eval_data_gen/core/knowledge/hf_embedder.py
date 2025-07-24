# src/eval_data_gen/core/knowledge/hf_embedder.py

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class HFEmbedder:
    """
    Minimal HuggingFace-based embedder for long texts.
    - Uses Snowflake (or any other HF model) directly.
    - No SentenceTransformer dependency.
    - No truncation (we rely on model's own max length).
    - Returns NumPy float32 vectors (ready for FAISS).
    """

    def __init__(
        self,
        model_name: str = "snowflake/snowflake-arctic-embed-l",
        device: str | None = None,
        normalize: bool = True,
    ):
        """
        Parameters
        ----------
        model_name : str
            HF model hub name (default: Snowflake Arctic embed large).
        device : str | None
            "cuda" or "cpu". If None, picks automatically.
        normalize : bool
            If True, L2-normalize the output vectors (good for cosine/IP).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        logger.info("HFEmbedder loaded '%s' on %s", model_name, self.device)

    @torch.no_grad()
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """
        Embed a list of texts and return a 2D NumPy array [num_texts, dim].

        Steps:
        1. Tokenize with truncation=False (do not cut).
        2. Forward pass through model.
        3. Masked mean-pool token embeddings to get one vector per text.
        4. (Optional) L2-normalize.
        """
        vectors: List[np.ndarray] = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # [1, seq_len, 1]

            summed = (last_hidden * mask).sum(dim=1)  # [1, hidden_dim]
            counts = mask.sum(dim=1).clamp(min=1e-9)  # [1, 1]
            pooled = (summed / counts).squeeze(0)     # [hidden_dim]

            vec = pooled.cpu().numpy().astype("float32")
            if self.normalize:
                # L2 normalize
                denom = np.linalg.norm(vec) + 1e-12
                vec = (vec / denom).astype("float32")

            vectors.append(vec)

        return np.vstack(vectors)