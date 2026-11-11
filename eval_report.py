"""
eval_report.py: run after build_index.py. Prints a markdown table of what BM25,
Dense and Hybrid retrieve for your 5 test questions, so you can paste it into
the README report and write the "best mode and why" column yourself.

Usage:  python eval_report.py
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

QUESTIONS = [
    "What is Student Help Desk?",                                   # exact rare keyword (BM25 should win)
    "How do I make my apps trigger each other automatically?",      # paraphrase, shares few words (dense should win)
    "Which certificate did you get for typesetting documents?",     # paraphrase of the LaTeX workshop
    "Do you know how to work with APIs?",                           # mixed
    "What is your favorite movie?",                                 # not in notes (should say don't know)
]
TOP_K = 3
RRF_K = 60

chunks = json.load(open("chunks.json", encoding="utf-8"))
emb = np.load("embeddings.npy")
tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_bm25(q):
    s = bm25.get_scores(tok(q))
    return [(int(i), float(s[i])) for i in np.argsort(-s)]


def rank_dense(q):
    s = emb @ model.encode([q], normalize_embeddings=True)[0]
    return [(int(i), float(s[i])) for i in np.argsort(-s)]


def rrf(*rankings):
    f = {}
    for r in rankings:
        for pos, (i, _) in enumerate(r, 1):
            f[i] = f.get(i, 0) + 1 / (RRF_K + pos)
    return sorted(f.items(), key=lambda x: -x[1])


def fmt(r):
    return "<br>".join(f"{chunks[i]['chunk_id']} ({s:.3f})" for i, s in r[:TOP_K])


print("| Question | BM25 top-3 | Dense top-3 | Hybrid (RRF) top-3 | Best mode and why |")
print("|---|---|---|---|---|")
for n, q in enumerate(QUESTIONS, 1):
    b, d = rank_bm25(q), rank_dense(q)
    print(f"| Q{n}. {q} | {fmt(b)} | {fmt(d)} | {fmt(rrf(b, d))} | *(write this yourself)* |")
