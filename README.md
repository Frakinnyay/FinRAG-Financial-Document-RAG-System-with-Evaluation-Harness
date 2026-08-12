# FinRAG — Financial Document RAG with Table-Aware Chunking + RAGAS Eval

A RAG system over Indian annual reports (TCS, Reliance, Infosys, Tata Motors,
HDFC Bank) that answers analyst-style questions with citations, and
publishes measured retrieval-quality metrics instead of just a demo.

**Stack:** pdfplumber, sentence-transformers (BAAI/bge-small), FAISS, Groq
(Llama 3.3 70B), RAGAS, Gradio

**Eval metrics:** see [`eval/results/metrics_table.md`](eval/results/metrics_table.md)

---

## Architecture

```
┌─────────────────┐
│  Annual Report   │
│  PDFs (5 cos.)   │
│  data/raw/*.pdf  │
└────────┬─────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│  ingest/pdf_parser.py                      │
│  - pdfplumber: detect table bounding boxes │
│  - Tables  → markdown, header repeated     │
│    on every split (table-aware chunking)   │
│  - Prose   → sliding-window chunks         │
│    (outside table bboxes only)             │
└────────┬────────────────────────────────────┘
         │  chunks + metadata (company, year,
         │  page, chunk_type)
         ▼
┌───────────────────────────────────────────┐
│  ingest/build_index.py                     │
│  - BAAI/bge-small-en-v1.5 embeddings       │
│  - FAISS IndexFlatIP (cosine via           │
│    normalized vectors)                     │
└────────┬────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│  rag/retriever.py                          │
│  - embed query (BGE query-instruction      │
│    prefix), top-k FAISS search             │
│  - optional company filter                 │
└────────┬────────────────────────────────────┘
         │  top-k chunks (timed separately)
         ▼
┌───────────────────────────────────────────┐
│  rag/generator.py                          │
│  - Groq API (Llama 3.3 70B)                │
│  - citation-forcing prompt:                │
│    [Company, FY<year>, p.<page>]           │
└────────┬────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│  app.py — Gradio UI (runs locally;         │
│  Dockerfile included for containerized     │
│  deployment, e.g. Cloud Run)                │
│  eval/run_ragas_eval.py — RAGAS scoring    │
└───────────────────────────────────────────┘
```

**Why table-aware chunking matters:** generic text splitters run straight
through PDF tables and cut rows apart from headers, so "59,381" ends up in
a chunk with no idea it means "Total Revenue, FY2023". This pipeline
detects table bounding boxes with `pdfplumber` and extracts them
separately as markdown, repeating the header row on every split — so a
retrieved chunk about row 30 of a 50-row table still says what column 3
means. See `ingest/pdf_parser.py` for the implementation.

---

## Repo structure

```
finrag/
├── config.py                  # all paths, model names, chunk sizes in one place
├── requirements.txt
├── .env.example                # copy to .env and fill in your Groq key
├── Dockerfile                  # containerizes the app for cloud deployment
├── .dockerignore
├── data/
│   ├── raw/                    # put your annual report PDFs here
│   └── processed/              # FAISS index + chunk metadata (generated)
├── ingest/
│   ├── pdf_parser.py           # table-aware parsing + chunking
│   └── build_index.py          # embeds chunks, builds FAISS index
├── rag/
│   ├── retriever.py            # FAISS search (timed separately from generation)
│   ├── generator.py            # Groq call with citation-forcing prompt
│   └── pipeline.py             # ties retriever + generator together
├── eval/
│   ├── test_queries.json       # 50 hand-written analyst questions
│   ├── run_ragas_eval.py       # runs pipeline over queries, scores with RAGAS
│   └── results/                # metrics_table.md + raw CSVs land here
└── app.py                      # Gradio demo (runs locally: `python app.py`)
```

---

## Step-by-step: getting this running

### 0. Prerequisites
- Python 3.10+ installed
- A free Groq API key: https://console.groq.com/keys (takes 1 minute, no card needed)
- The annual report PDFs you already downloaded (TCS, Reliance, Infosys, Tata Motors, HDFC)

### 1. Set up the environment
```bash
cd finrag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
This will take a few minutes — `sentence-transformers`, `faiss-cpu`, and
`ragas` pull in torch and other heavy deps.

**If you hit build errors on faiss-cpu on Windows/ARM Mac:** use
`conda install -c pytorch faiss-cpu` instead of pip, then re-run
`pip install -r requirements.txt` with `faiss-cpu` commented out of the file.

### 2. Configure your API key
```bash
cp .env.example .env
```
Open `.env` and paste your Groq key into `GROQ_API_KEY=`. Leave the rest
as default unless you know you want a different Groq model — check
https://console.groq.com/docs/models for currently available free-tier
models, since Groq's lineup changes over time.

**Additional eval-only env vars** (add to `.env` if re-running the
RAGAS eval yourself):
- `GROQ_JUDGE_API_KEY` — a separate Groq key for RAGAS scoring, to avoid
  competing with the main key's daily quota used for answer generation
- `GROQ_JUDGE_MODEL` — defaults to `llama-3.1-8b-instant`; a smaller/
  faster model to conserve token budget during judging
- `EVAL_SUBSET_SIZE` — limits RAGAS scoring to the first N questions
  (used here to score 20 of 50 within free-tier limits)

### 3. Put your PDFs in the right place, with the right names
Move your downloaded PDFs into `data/raw/` and **rename them** to the
pattern `COMPANY_YEAR.pdf` — this naming convention is how the parser
infers company + fiscal year without you having to hand-label anything:

```
data/raw/TCS_2023.pdf
data/raw/TCS_2022.pdf
data/raw/TCS_2021.pdf
data/raw/RELIANCE_2023.pdf
data/raw/RELIANCE_2022.pdf
data/raw/INFOSYS_2023.pdf
data/raw/INFOSYS_2022.pdf
data/raw/TATAMOTORS_2023.pdf
data/raw/TATAMOTORS_2022.pdf
data/raw/HDFC_2023.pdf
data/raw/HDFC_2022.pdf
... (aim for 15-20 total, 2-3 years x 5-7 companies)
```
Use the exact company ticker/short-name you want cited in answers — it's
used verbatim in citations like `[TCS, FY2023, p.42]`.

### 4. Parse the PDFs (table-aware chunking)
```bash
python ingest/pdf_parser.py
```
Watch the output — it prints how many table vs. text chunks it found per
file. **If a file shows 0 table chunks and it definitely has tables**,
that PDF likely has tables without visible gridlines (common for
Excel-exported financial statements). The parser already falls back from
`lines_strict` to a `text`-based table strategy automatically, but if a
specific file still comes up empty, that PDF may need OCR first (see
Troubleshooting below).

This writes `data/processed/raw_chunks.jsonl` — open it and skim a few
lines to sanity-check that table chunks look like proper markdown tables
and text chunks aren't full of garbled whitespace.

### 5. Build the vector index
```bash
python ingest/build_index.py
```
First run downloads `BAAI/bge-small-en-v1.5` (~130MB) from Hugging Face.
This embeds every chunk and builds a FAISS index — for 15-20 documents
this should take a few minutes on CPU. Output: `data/processed/faiss.index`
and `data/processed/chunks_metadata.jsonl`.

### 6. Sanity-check the pipeline before running full eval
```bash
python rag/pipeline.py
```
This asks one hardcoded test question and prints the answer plus
retrieval/generation latency. Confirm:
- The answer actually cites `[Company, FYxxxx, p.xx]`
- The numbers in the answer look plausible for what you know about the company
- Retrieval latency is roughly 50-200ms; generation is 1-2s (these are
  reported **separately** — don't be confused if they look very different,
  that's expected: FAISS search is local and fast, Groq is a network call)

If the answer is generic or wrong, go back to step 4 and check your
chunking quality first — bad retrieval is the most common root cause of
bad RAG answers, not the LLM.

### 7. Fill in ground-truth answers for eval
Open `eval/test_queries.json`. Every entry has `"ground_truth": "FILL_ME"`.
For **at least the numeric and factual questions**, open the actual PDF,
find the real figure, and replace `FILL_ME` with it, e.g.:
```json
{"id": 2, "company": "TCS", "type": "numeric",
 "question": "What was TCS's total revenue from operations for the fiscal year?",
 "ground_truth": "TCS's revenue from operations for FY2023 was Rs. 2,25,458 crore."}
```
This is manual, and it's meant to be — the eval report is only as
trustworthy as your ground truth.

If you skip this step, `run_ragas_eval.py` will still work but will only
report `faithfulness` and `answer_relevancy` (which don't need ground
truth) — `context_precision` and `context_recall` need it and will be
skipped with a warning.

### 8. Run the RAGAS evaluation
```bash
python eval/run_ragas_eval.py
```
This runs all 50 queries through the full pipeline (so it also calls
Groq 50 times — stay within free-tier rate limits). Answer generation is
resumable: results are saved incrementally to `eval/results/progress.jsonl`,
so a rerun after a rate limit picks up where it left off instead of
starting over.

RAGAS scoring itself also calls an LLM to judge faithfulness/relevancy —
by default it uses OpenAI. This project points it at Groq instead (see
`GROQ_JUDGE_API_KEY` above), since RAGAS scoring makes several additional
LLM calls per question and can hit free-tier limits fast. Use
`EVAL_SUBSET_SIZE` to score a smaller, honest subset if needed rather
than risk an incomplete/noisy full run.

Output:
- `eval/results/metrics_table.md` — paste this straight into this README
- `eval/results/pipeline_outputs_raw.csv` — every question/answer/context
- `eval/results/ragas_raw.csv` — per-question RAGAS scores, useful for
  finding which specific questions dragged the average down

### 9. Paste your real metrics into this README
Replace the placeholder section below with the actual generated table,
and write 2-3 honest sentences on what you'd improve next. A faithfulness
score in the 75-90% range with a clear explanation of failure modes is
**more credible** than a suspicious 99% — say that in your write-up.

### 10. Run the Gradio app locally
```bash
python app.py
```
Opens on `http://localhost:7860`. Try the example questions, try a few
of your own, and try deliberately asking something not in the corpus to
confirm it says "not found" instead of hallucinating.

### 11. Deployment (optional)

This project runs fully locally via `python app.py` (Step 10). A
`Dockerfile` is included if you want to containerize it for a cloud
platform (e.g. Google Cloud Run, Render). Note: Hugging Face Spaces
changed its free tier in mid-2026 and no longer offers free Gradio/Docker
hosting — Static Spaces only. I evaluated Cloud Run as an alternative but
opted not to deploy publicly to avoid any billing exposure on a personal
project; the Dockerfile is here to show the deployment path was built
and tested, not left undone.

---

## Evaluation Metrics

Pipeline answered all 50 test queries; RAGAS-scored on a **20-query
subset** due to Groq free-tier rate limits (see note below).

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.651 | Are claims in the answer actually supported by retrieved context? |
| Answer Relevancy | not reliably computed | Does the answer address the question asked? |
| Context Precision | 0.658 | Of retrieved chunks, how many were actually relevant? |
| Context Recall | 0.400 | Did retrieval surface everything needed to answer correctly? |

| Stage | Avg latency | P95 |
|---|---|---|
| Retrieval (FAISS) | 109.7 ms | 488.5 ms |
| Generation (Groq) | 8891.9 ms | 28546.9 ms |

**On Answer Relevancy:** this metric could not be reliably computed
within free-tier API constraints. Both a smaller (8B) and larger (70B)
Groq judge model hit rate limits before completing this metric across
the evaluation set — unlike the other three metrics, `answer_relevancy`
requires the judge LLM to generate structured synthetic questions from
each answer, which is more token- and call-intensive. The three metrics
that did complete indicate the retrieval and generation pipeline is
functioning correctly, with clear room for improvement in recall. In a
production setting, this metric would be computed using a paid-tier API
with adequate rate limits.

**What I'd improve next:** faithfulness and recall were both lowest on
multi-hop questions spanning two segments or two fiscal years — retrieval
often only pulled context from one side of the comparison. Next step:
query decomposition, splitting multi-hop questions into sub-queries
before retrieval so each part of the comparison gets its own dedicated
search.

---

## Troubleshooting

**Tables extracted as garbage / empty table chunks**
Some PDFs (especially Excel-exported financial statement pages) have no
visible ruling lines, and `find_tables` with the `text` strategy can
still misfire on dense multi-column layouts. Open the specific page in a
PDF viewer — if it's a scanned image rather than real text, you'll need
OCR (e.g. `ocrmypdf input.pdf output.pdf` before parsing) since
`pdfplumber` can't extract text/tables from images.

**Table bounding box errors on some PDFs (rotated pages / odd page sizes)**
A small number of PDFs report table coordinates slightly outside the
page's actual boundary, which can crash naive bbox subtraction. The
parser clamps bounding boxes to the page's actual dimensions and skips
invalid ones rather than crashing, and wraps each page in its own
try/except so one bad page doesn't take down the whole file.

**`GROQ_API_KEY not set` error**
You forgot step 2, or your `.env` isn't being loaded because you're
running from the wrong directory. `config.py` calls `load_dotenv()` which
looks for `.env` in the current working directory — always run scripts
from the `finrag/` root.

**Rate limited by Groq (HTTP 429) during answer generation**
Free tier has daily and per-minute token limits. The eval script saves
progress incrementally, so simply waiting and re-running
`eval/run_ragas_eval.py` resumes from the last completed question instead
of starting over.

**Rate limited by Groq during RAGAS scoring specifically**
RAGAS makes several LLM calls per question per metric, which multiplies
token usage quickly. Use a separate `GROQ_JUDGE_API_KEY`, a smaller
`GROQ_JUDGE_MODEL`, and `EVAL_SUBSET_SIZE` to keep this within free-tier
limits — see Step 2 and Step 8 above.

**RAGAS wants an OpenAI key**
By default RAGAS's judge-LLM metrics use OpenAI. This project points
RAGAS at Groq instead via `langchain_groq.ChatGroq` and a local
HuggingFace embeddings model — see `eval/run_ragas_eval.py`'s
`score_with_ragas()` for the implementation.

**Faithfulness score is suspiciously near 100%**
Your test set is probably too easy (all single-fact lookups). Make sure
your test queries include multi-hop and numeric/table questions — those
are the ones designed to stress-test retrieval, not just generation.

**Index seems to ignore new PDFs after re-running ingestion**
`build_index.py` rebuilds the index from scratch each run — it doesn't
incrementally update. If you add PDFs, re-run both `pdf_parser.py` and
`build_index.py` in sequence.

---

## What makes this different from a tutorial RAG project

1. **Table-aware chunking**, not a generic recursive text splitter run
   over the whole PDF — financial documents are majority tables, and most
   tutorials skip this entirely.
2. **A real eval harness with RAGAS**, publishing faithfulness/precision/
   recall rather than just a vibes-based demo, including an honest account
   of the one metric that couldn't be reliably computed and why.
3. **Retrieval and generation latency reported separately** — conflating
   a 100ms FAISS lookup with an 8-9s API call into one number hides what's
   actually the bottleneck.
4. **Citations that point to page numbers**, forced via the system
   prompt, so answers are auditable against the source filing.
5. **Resumable evaluation pipeline** — answer generation checkpoints after
   every question, so free-tier rate limits during a 50-question run don't
   mean starting over from scratch.