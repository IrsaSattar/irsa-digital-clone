"""
build_index.py: run this ONCE on your laptop (or Colab) before deploying.

Steps:
  1. Load every .md / .txt file in data/
  2. Split into overlapping chunks (~250 words, 40-word overlap)
  3. Compute a sentence embedding for each chunk
  4. Save chunks.json (text + source + chunk_id) and embeddings.npy

Usage:  python build_index.py
"""
import json
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
CHUNK_WORDS = 220      # target chunk size (assignment: 200-300 words)
OVERLAP_WORDS = 40     # small overlap between neighbouring chunks
MODEL_NAME = "all-MiniLM-L6-v2"


def split_into_chunks(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Sliding window over words. Returns a list of chunk strings."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = chunk_words - overlap
    while start < len(words):
        piece = words[start:start + chunk_words]
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
        start += step
    # merge a tiny last chunk into the previous one so we never keep a stub
    if len(chunks) > 1 and len(chunks[-1].split()) < 60:
        tail = chunks.pop().split()
        prev = chunks.pop().split()
        # only add words not already covered by the overlap
        chunks.append(" ".join(prev + tail[overlap:]))
    return chunks


def load_chunks():
    records = []
    files = sorted(list(DATA_DIR.glob("*.md")) + list(DATA_DIR.glob("*.txt")))
    for path in files:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[ \t]+", " ", text)
        for i, chunk in enumerate(split_into_chunks(text)):
            records.append({
                "chunk_id": f"{path.stem}-{i}",
                "source": path.name,
                "text": chunk,
            })
    return records


def main():
    records = load_chunks()
    print(f"{len(records)} chunks from {len({r['source'] for r in records})} files")
    for r in records:
        print(f"  {r['chunk_id']:<40} {len(r['text'].split())} words")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,   # unit vectors -> dot product = cosine
        show_progress_bar=True,
    )
    np.save("embeddings.npy", emb.astype("float32"))
    Path("chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved embeddings.npy and chunks.json")


if __name__ == "__main__":
    main()
