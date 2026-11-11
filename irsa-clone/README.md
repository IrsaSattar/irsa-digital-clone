# 🤖 Irsa Sattar's Digital Clone (Hybrid RAG)

> **This is an AI clone, not the real person.** It answers only from notes that Irsa Sattar wrote about herself.

A chatbot that answers questions as me, using a Retrieval-Augmented Generation (RAG) pipeline:

1. **Chunk**: 8 documents are split into ~220-word chunks with a 40-word overlap (source file name and chunk ID kept).
2. **Index twice**: BM25 (`rank_bm25`) and sentence embeddings (`all-MiniLM-L6-v2`, precomputed and saved as `embeddings.npy`).
3. **Retrieve**: BM25 only, Dense only (cosine similarity with numpy, no vector database), or Hybrid.
4. **Fuse**: Hybrid uses **Reciprocal Rank Fusion**, `score = Σ 1 / (60 + rank)`.
5. **Generate**: an LLM answers in first person, only from the retrieved context, and says *"I don't have that in my notes."* when the answer is missing.
6. **Show your work**: the app displays the answer plus every retrieved chunk with its source, chunk ID, score and BM25/dense ranks.

Built with Streamlit. The API key is stored as a Streamlit secret and is never in the code or the repo.

---

# One-Page Report

## 1. Knowledge base and why I chose it

I used myself as the topic, so I can judge every answer immediately. The knowledge base has 8 Markdown files (about 4,400 words, 25 chunks):

| File | Type | Purpose |
|---|---|---|
| `bio.md` | Factual | Who I am, education, interests |
| `cv_skills.md` | Factual | Skills, certifications, projects |
| `projects.md` | Factual | Overview of my two projects |
| `student_help_desk_writeup.md` | Long-form | Detailed write-up of my RAG project |
| `automation_notes.md` | Long-form (study notes) | n8n, webhooks, REST APIs, JSON |
| `work_style_and_values.md` | Personality / opinions | How I work, values, opinions, advice |
| `learning_and_certifications.md` | Factual / narrative | Learning path and certifications |
| `faq.md` | First-person FAQ | 28 questions answered in my own voice |

Rare keywords for BM25 to win on: **"Student Help Desk"**, **"n8n"**, **"LaTeX"**, **"webhook"**. Plain descriptive passages (for example, "letting software do repetitive tasks by itself") give dense search something to win on paraphrased questions. No phone numbers, addresses, ID numbers or other private data are included.

## 2. Test questions: BM25 vs Dense vs Hybrid

Generate this table with `python eval_report.py` (after `python build_index.py`) and paste it here, then write the last column yourself.
| Question | BM25 top-3 | Dense top-3 | Hybrid (RRF) top-3 | Best mode and why |
|---|---|---|---|---|
| Q1. What is Student Help Desk? | projects-0 (4.300)<br>faq-0 (3.937)<br>student_help_desk_writeup-0 (3.897) | student_help_desk_writeup-0 (0.388)<br>student_help_desk_writeup-3 (0.361)<br>projects-0 (0.356) | projects-0 (0.032)<br>student_help_desk_writeup-0 (0.032)<br>student_help_desk_writeup-3 (0.031) | *(Best mode and why)* |
| Q2. How do I make my apps trigger each other automatically? | bio-0 (9.652)<br>automation_notes-0 (5.900)<br>faq-2 (5.317) | bio-0 (0.182)<br>automation_notes-0 (0.132)<br>automation_notes-1 (0.123) | bio-0 (0.033)<br>automation_notes-0 (0.032)<br>faq-2 (0.030) | *(write this yourself)* |
| Q3. Which certificate did you get for typesetting documents? | learning_and_certifications-0 (5.368)<br>faq-2 (4.902)<br>learning_and_certifications-1 (4.010) | learning_and_certifications-0 (0.392)<br>faq-2 (0.200)<br>cv_skills-1 (0.154) | learning_and_certifications-0 (0.033)<br>faq-2 (0.032)<br>learning_and_certifications-1 (0.031) | *(write this yourself)* |
| Q4. Do you know how to work with APIs? | faq-2 (7.901)<br>faq-3 (5.460)<br>faq-1 (4.773) | automation_notes-2 (0.570)<br>faq-2 (0.521)<br>cv_skills-0 (0.468) | faq-2 (0.033)<br>faq-3 (0.030)<br>learning_and_certifications-1 (0.030) | *(write this yourself)* |
| Q5. What is your favorite movie? | faq-4 (6.213)<br>faq-0 (4.717)<br>work_style_and_values-2 (3.271) | bio-2 (0.127)<br>projects-1 (0.116)<br>work_style_and_values-1 (0.096) | faq-4 (0.032)<br>faq-0 (0.031)<br>projects-1 (0.031) | *(write this yourself)* |

| Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|
| Q1. What is Student Help Desk? *(exact rare keyword)* | | | | |
| Q2. How do I make my apps trigger each other automatically? *(paraphrase)* | | | | |
| Q3. Which certificate did you get for typesetting documents? *(paraphrase of LaTeX)* | | | | |
| Q4. Do you know how to work with APIs? | | | | |
| Q5. What is your favorite movie? *(not in notes)* | | | | |

**Fusion method.** I implemented Reciprocal Rank Fusion because it only uses ranks, so it needs no score normalization (BM25 scores are unbounded while cosine scores are between -1 and 1, so they can't be added directly). The alternative is a weighted score `alpha * dense + (1 - alpha) * bm25` after min-max normalizing each score list; it gives control over which retriever matters more, but needs tuning of `alpha`.

## 3. Failure case

*(Fill in after you test the app. See the tips below.)*

Real observation I measured while testing BM25 on the chunks: for the question *"how do you like to study new things"*, BM25 ranked the **FAQ** chunk first, because it contains many common words like "do", "you" and "like" repeated in every question. The real answer lives in `work_style_and_values.md` ("How I learn"). So BM25 rewards generic, repeated words in FAQ-style text and can push the better chunk down. Write your own failure case in this section from an actual wrong or weak answer from the live clone, and explain why it happened (wrong chunk retrieved, chunk too broad, or the LLM adding something that was not in the context).

## 4. One improvement I would make next

Add a cross-encoder reranker on top of the hybrid results, and remove stopwords in the BM25 tokenizer. The reranker would read the question and each candidate chunk together, which fixes cases where the right chunk is retrieved at rank 3 or 4 but not ranked first. I would also add a small hand-labeled evaluation set with hit@3 for each mode, so improvements can be measured instead of judged by eye.

---

# Setup and deployment (Streamlit Community Cloud)

### A. Run once on your laptop

```bash
pip install -r requirements.txt
python build_index.py        # creates chunks.json and embeddings.npy
python eval_report.py        # prints the table for the report
```

### B. Test locally (optional)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key
streamlit run app.py
```

### C. Deploy

1. Create a **public** GitHub repository and upload everything in this folder, **including** `chunks.json` and `embeddings.npy`. Do **not** upload `.streamlit/secrets.toml` (it is in `.gitignore`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose your repo, branch `main` and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste one of these (your real key inside the quotes):
   ```toml
   GROQ_API_KEY = "..."
   ```
   or `GEMINI_API_KEY = "..."` or `HF_TOKEN = "..."`.
4. Click **Deploy**. Submit the public app link (and the GitHub repo link) as your assignment link.

If the provider renames its default model, add `LLM_MODEL = "new-model-name"` to the secrets.

### Files

```
app.py                  Streamlit app: BM25, dense, RRF, generation, UI
build_index.py          chunking + embeddings (run once)
eval_report.py          prints the 5-question comparison table
requirements.txt
chunks.json             generated by build_index.py
embeddings.npy          generated by build_index.py
data/                   the 8 knowledge-base documents
.streamlit/secrets.toml.example
```
