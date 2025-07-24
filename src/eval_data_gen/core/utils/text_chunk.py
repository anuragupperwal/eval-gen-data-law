# text_chunk.py
import nltk
from typing import List

# One-time download (noop if already present)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def sent_merge(text: str, target_tokens: int = 256) -> List[str]:
    """
    Split text into sentences, then merge contiguous sentences until
    ≈ target_tokens words.  Returns longer, natural-boundary passages.
    """
    sentences = nltk.sent_tokenize(text)
    buff, cur, out = [], 0, []
    for s in sentences:
        tok = s.split()
        if cur + len(tok) > target_tokens and buff:
            out.append(" ".join(buff))
            buff, cur = [], 0
        buff.append(s)
        cur += len(tok)
    if buff:
        out.append(" ".join(buff))
    return out