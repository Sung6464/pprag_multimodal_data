# Proxy-Pointer MultiModal (Gemini-only recreation)

A self-contained recreation of the **MultiModal** part of
[Proxy-Pointer-RAG](https://github.com/Proxy-Pointer/Proxy-Pointer-RAG) —
structure-aware retrieval that answers with text **and** the actual
figures/tables it relies on.

This build differs from the original in three ways you asked for:

1. **MultiModal only.** No Text-Only and no DocComparator.
2. **Gemini key only — no Adobe.** The original used the Adobe Extract PDF API.
   This version extracts everything locally with **PyMuPDF** (PDF) and
   **python-docx** (DOCX), so a single `GOOGLE_API_KEY` is all you need.
3. **Drag-and-drop in the UI + a visible proxy-pointer ledger.** You can drop
   new PDF/DOCX files straight into the Streamlit UI to extract, structure,
   embed and index them, and every structural pointer ("proxy") is written to a
   separate file you can view and download in the app.

---

## How it works

```
Upload (PDF/DOCX)
   │  PyMuPDF / python-docx
   ▼
Markdown + figures/  ──►  Structure tree (Section > Sub-section)   each node =
   │                       with image anchors                       a proxy pointer
   │                                                                      │
   │                                                          append to ledger file
   ▼                                                          data/proxies/proxy_pointers.jsonl
Embed nodes (gemini-embedding-001, 1536-d)  ──►  FAISS cosine index
                                                       │
Query ─► embed ─► recall k=200 ─► dedup ─► re-rank (Gemini, anchor-aware) ─► top 5
                                                       │
                       load full section text + real images
                                                       │
        Gemini 3.1 Flash-Lite multimodal synthesis ─► answer + [SHOW: img | caption]
                                                       │
                                   Streamlit renders the cited figures inline
```

A "proxy pointer" is the structural breadcrumb of a section, e.g.
`CLIP > 3 Method > 3.1 Discriminative Visual-Text Alignment`. The system indexes
these pointers (not blind text chunks) and loads the **full** section + its
images as the payload at answer time.

---

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env -> GOOGLE_API_KEY=...   (get one at https://aistudio.google.com/apikey)
```

## Run

```bash
streamlit run app.py
```

Then in the browser:

1. **Sidebar → Add documents:** drop one or more `.pdf` / `.docx` files and
   click **Process & index**. Extraction is local; embedding/indexing uses your
   Gemini key.
2. **Ask tab:** type a question. You get a grounded answer, the figures/tables
   it cites rendered inline, and the structural pointers used as sources.
3. **Proxy Pointers tab:** see every pointer created so far (newest first),
   filter by document, inspect a pointer's snippet and image anchors, and
   download the raw ledger.

---

## Where things are stored

| Path | Contents |
| --- | --- |
| `data/extracted_papers/<doc>/<doc>.md` | extracted markdown |
| `data/extracted_papers/<doc>/figures/` | extracted images |
| `data/trees/<doc>.tree.json` | the structure tree |
| `data/index/faiss.index`, `meta.json` | vector index + metadata |
| `data/proxies/proxy_pointers.jsonl` | **the proxy-pointer ledger** (one JSON record per line) |

Re-processing a document replaces its old markdown, tree, vectors, and ledger
entries, so you never get duplicates.

---

## Configuration

All knobs live in `pp/config.py` and can be overridden via environment
variables (see `.env.example`): models (`PP_GEN_MODEL`, `PP_EMBED_MODEL`,
`PP_EMBED_DIM`), retrieval (`PP_RECALL_K`, `PP_CANDIDATE_K`, `PP_FINALIST_K`),
and storage (`PP_DATA_DIR`).

Defaults: generation/vision = `gemini-3.1-flash-lite`, embeddings =
`gemini-embedding-001` at 1536 dimensions.

---

## Notes & limits

* Local PDF heading detection is heuristic (font-size based). Clean, well-
  structured PDFs produce the best trees; very unusual layouts may yield a
  flatter tree — retrieval still works, the breadcrumbs are just shallower.
* Scanned/image-only PDFs have no extractable text; add OCR upstream if needed.
* The synthesizer only shows images that exist in the retrieved sections, so it
  cannot invent a figure.

Original architecture & write-ups: see the upstream repo's README and the
linked Towards Data Science articles.
