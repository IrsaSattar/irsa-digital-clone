"""
Irsa Sattar's Digital Clone: Hybrid RAG (BM25 + dense + RRF) on Streamlit.

Pipeline: load chunks -> BM25 index + precomputed embeddings -> retrieve
(BM25 / Dense / Hybrid) -> LLM answers in first person, only from context.
"""
import json
import os
import re

import numpy as np
import requests
import streamlit as st
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CHUNKS_PATH = "chunks.json"
EMB_PATH = "embeddings.npy"
EMB_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60  # constant in score = sum(1 / (60 + rank))

PERSONA_PROMPT = """You are the AI clone of Irsa Sattar, a 7th-semester BSCS student at the University of Management and Technology (UMT) who loves AI and automation.
Speak in the first person ("I", "my"), in a casual and friendly tone. Keep answers short and natural, with an occasional emoji.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of Irsa Sattar.

CONTEXT:
{context}

QUESTION:
{question}"""


# ----------------------------------------------------------------------------
# Secrets: works with Streamlit secrets (cloud) or environment variables (local)
# ----------------------------------------------------------------------------
def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


# ----------------------------------------------------------------------------
# Load data and build indexes (cached, so this runs once per server start)
# ----------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMB_PATH)  # precomputed by build_index.py
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return chunks, embeddings, bm25


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)


# ----------------------------------------------------------------------------
# Retrievers: each returns a full ranking as a list of (chunk_index, score)
# ----------------------------------------------------------------------------
def bm25_rank(query, bm25):
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]


def dense_rank(query, embeddings):
    q = load_embedder().encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q  # unit vectors, so dot product = cosine similarity
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]


def rrf_fuse(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank(d)), rank starts at 1."""
    fused = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def retrieve(query, mode, top_k, chunks, embeddings, bm25):
    """Returns a list of result dicts for the top_k chunks."""
    b = bm25_rank(query, bm25)
    d = dense_rank(query, embeddings)
    b_rank = {i: r for r, (i, _) in enumerate(b, 1)}
    d_rank = {i: r for r, (i, _) in enumerate(d, 1)}
    b_score = dict(b)
    d_score = dict(d)

    if mode == "BM25 only":
        ranked = b
    elif mode == "Dense only":
        ranked = d
    else:
        ranked = rrf_fuse([b, d])

    results = []
    for idx, score in ranked[:top_k]:
        results.append({
            "chunk": chunks[idx],
            "score": score,
            "bm25_score": b_score[idx], "bm25_rank": b_rank[idx],
            "dense_score": d_score[idx], "dense_rank": d_rank[idx],
        })
    return results


# ----------------------------------------------------------------------------
# Generation: Groq, Gemini or Hugging Face (whichever key you put in secrets)
# ----------------------------------------------------------------------------
def build_prompt(question, results):
    context = "\n\n".join(
        f"[Source: {r['chunk']['source']}]\n{r['chunk']['text']}" for r in results
    )
    return PERSONA_PROMPT.format(context=context, question=question)


def call_openai_compatible(url, key, model, prompt):
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_gemini(key, model, prompt):
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_answer(question, results):
    """Returns (answer, note). note explains fallback mode if no LLM is available."""
    prompt = build_prompt(question, results)
    model_override = get_secret("LLM_MODEL")

    try:
        if get_secret("GROQ_API_KEY"):
            return call_openai_compatible(
                "https://api.groq.com/openai/v1/chat/completions",
                get_secret("GROQ_API_KEY"),
                model_override or "llama-3.1-8b-instant",
                prompt,
            ), None
        if get_secret("GEMINI_API_KEY"):
            return call_gemini(
                get_secret("GEMINI_API_KEY"),
                model_override or "gemini-2.0-flash",
                prompt,
            ), None
        if get_secret("HF_TOKEN"):
            return call_openai_compatible(
                "https://router.huggingface.co/v1/chat/completions",
                get_secret("HF_TOKEN"),
                model_override or "meta-llama/Llama-3.1-8B-Instruct",
                prompt,
            ), None
    except Exception as e:  # network error, quota, wrong model name...
        return templated_answer(results), f"LLM call failed ({type(e).__name__}); showing templated answer."

    return templated_answer(results), "No API key found; showing templated answer."


def templated_answer(results):
    top = results[0]["chunk"]
    return f"Here's the closest thing in my notes (from {top['source']}):\n\n{top['text']}"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Irsa's AI Clone", page_icon="🤖", layout="wide")

st.warning("⚠️ **This is an AI clone, not the real person.**  It answers only from notes Irsa Sattar wrote about herself.")
st.title("🤖 Chat with Irsa's Digital Clone")
st.caption("Hybrid RAG: BM25 keyword search + dense embedding search + Reciprocal Rank Fusion")

if not (os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH)):
    st.error("chunks.json / embeddings.npy not found. Run `python build_index.py` first.")
    st.stop()

chunks, embeddings, bm25 = load_index()

with st.sidebar:
    st.header("Retrieval settings")
    mode = st.radio("Retrieval mode", ["BM25 only", "Dense only", "Hybrid"], index=2)
    top_k = st.slider("Chunks sent to the LLM (top-k)", 1, 6, 4)
    st.markdown("---")
    st.caption(f"{len(chunks)} chunks from {len({c['source'] for c in chunks})} documents")
    st.caption("Hybrid = Reciprocal Rank Fusion, score = Σ 1/(60 + rank)")

examples = [
    "What is Student Help Desk?",
    "Which tools do you use to connect apps together automatically?",
    "How do you like to study new things?",
    "What's your favorite movie?",
]
st.write("Try one:")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Ask me anything about myself", key="question")

if st.button("Ask", type="primary") or (question and st.session_state.get("_last") != (question, mode, top_k)):
    if question.strip():
        st.session_state["_last"] = (question, mode, top_k)
        with st.spinner("Retrieving and thinking..."):
            results = retrieve(question, mode, top_k, chunks, embeddings, bm25)
            answer, note = generate_answer(question, results)

        st.subheader("Answer")
        st.write(answer)
        if note:
            st.info(note)

        st.subheader(f"Retrieved chunks ({mode})")
        score_label = {"BM25 only": "BM25 score", "Dense only": "cosine similarity", "Hybrid": "RRF score"}[mode]
        for n, r in enumerate(results, 1):
            c = r["chunk"]
            title = f"#{n} · {c['source']} · {c['chunk_id']} · {score_label}: {r['score']:.4f}"
            with st.expander(title, expanded=(n == 1)):
                st.caption(
                    f"BM25: rank {r['bm25_rank']} (score {r['bm25_score']:.3f})  |  "
                    f"Dense: rank {r['dense_rank']} (cosine {r['dense_score']:.3f})"
                )
                st.write(c["text"])
