"""
MultiModal RAG bot.

Pipeline (mirrors the original Proxy-Pointer MultiModal design):
  1. Embed query, broad vector recall (k=RECALL_K)
  2. Dedup by (doc_id, node_id) -> top CANDIDATE_K unique candidates
  3. Anchor-aware semantic re-rank with 150-char snippets -> FINALIST_K nodes
  4. Load full section text + real image paths
  5. Gemini 3.1 Flash-Lite multimodal synthesis -> grounded answer that emits
     [SHOW: filename | caption] directives for the figures it actually uses
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import config, gemini
from .indexer import RetrievalStore

_ANCHOR_RE = re.compile(r"\b(figure|fig\.?|table|tab\.?|chart|eq(?:uation)?)\s*([0-9ivxIVX]+)", re.I)
_SHOW_RE = re.compile(r"\[SHOW:\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]")

RERANK_SYSTEM = (
    "You are a precise retrieval re-ranker for a structure-aware document RAG "
    "system. You are given a user query and a numbered list of candidate "
    "document sections, each shown as a breadcrumb pointer plus a short snippet. "
    "Pick the sections most likely to contain the answer. If the query names a "
    "specific Figure/Table/Equation, strongly prefer sections whose snippet or "
    "breadcrumb references that anchor. Respond with ONLY a JSON array of the "
    "chosen candidate numbers, best first, no prose."
)

SYNTH_SYSTEM = (
    "You are a meticulous multimodal research assistant. Answer the user's "
    "question using ONLY the provided document sections and the attached images. "
    "Ground every claim in the context. Be specific with numbers and names.\n\n"
    "When a provided image (figure/table/chart) supports your answer, cite it "
    "inline by emitting a directive on its own line in EXACTLY this format:\n"
    "[SHOW: <image_filename> | <short caption of what it shows>]\n"
    "Use the bare filename (e.g. img_4.png), not a path. Only SHOW images that "
    "appear in the provided context. If the context does not contain the answer, "
    "say so honestly rather than inventing one."
)


def _anchor_terms(query: str) -> set[str]:
    terms = set()
    for kind, num in _ANCHOR_RE.findall(query):
        terms.add(f"{kind.lower().rstrip('.')} {num.lower()}")
        terms.add(num.lower())
    return terms


class MMRagBot:
    def __init__(self):
        self.store = RetrievalStore()

    @property
    def ready(self) -> bool:
        return self.store.ready

    # -- step 1+2 --------------------------------------------------------
    def _recall(self, query: str) -> list[dict]:
        qvec = np.array([gemini.embed_query(query)], dtype="float32")
        import faiss

        faiss.normalize_L2(qvec)
        hits = self.store.search(qvec, config.RECALL_K)
        # dedup by (doc_id, node_id), keep best score
        seen: dict[tuple, dict] = {}
        for h in hits:
            key = (h["doc_id"], h["node_id"])
            if key not in seen:
                seen[key] = h
        cands = list(seen.values())[: config.CANDIDATE_K]
        return cands

    # -- step 3 ----------------------------------------------------------
    def _rerank(self, query: str, cands: list[dict]) -> list[dict]:
        if len(cands) <= config.FINALIST_K:
            return cands
        lines = []
        for i, c in enumerate(cands):
            img_note = f"  [contains {len(c['images'])} image(s)]" if c["images"] else ""
            lines.append(f"{i}. {c['breadcrumb']}{img_note}\n   snippet: {c['snippet']}")
        prompt = (
            f"User query: {query}\n\n"
            f"Candidate sections:\n" + "\n".join(lines) + "\n\n"
            f"Return a JSON array of the {config.FINALIST_K} best candidate numbers, best first."
        )
        try:
            raw = gemini.generate_text(prompt, system=RERANK_SYSTEM)
            nums = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            chosen = [cands[i] for i in nums if isinstance(i, int) and 0 <= i < len(cands)]
        except Exception:
            chosen = []

        # Anchor-aware safety net: ensure sections referencing a requested
        # Figure/Table anchor are included even if the LLM missed them.
        anchors = _anchor_terms(query)
        if anchors:
            for c in cands:
                hay = (c["snippet"] + " " + c["breadcrumb"]).lower()
                if any(a in hay for a in anchors) and c not in chosen:
                    chosen.insert(0, c)

        if not chosen:
            chosen = cands[: config.FINALIST_K]
        # de-dup preserve order
        out, seen = [], set()
        for c in chosen:
            key = (c["doc_id"], c["node_id"])
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out[: config.FINALIST_K]

    # -- step 4+5 --------------------------------------------------------
    def answer(self, query: str) -> dict:
        t0 = time.time()
        if not self.ready:
            return {"answer": "No documents are indexed yet. Upload a PDF or DOCX first.",
                    "sources": [], "images": [], "time_seconds": 0.0}

        cands = self._recall(query)
        finalists = self._rerank(query, cands)

        # Build text context + collect candidate image paths
        context_blocks = []
        image_paths: list[Path] = []
        filename_to_path: dict[str, Path] = {}
        for f in finalists:
            block = [f"### {f['breadcrumb']}", f.get("text", "")]
            if f["images"]:
                names = []
                for rel in f["images"]:
                    p = config.PAPERS_DIR / f["doc_id"] / rel
                    fname = Path(rel).name
                    filename_to_path[fname] = p
                    if p.exists() and p not in image_paths and len(image_paths) < 12:
                        image_paths.append(p)
                    names.append(fname)
                block.append(f"[Images available in this section: {', '.join(names)}]")
            context_blocks.append("\n".join(block))

        context = "\n\n---\n\n".join(context_blocks)
        prompt = (
            f"QUESTION:\n{query}\n\n"
            f"DOCUMENT SECTIONS (each begins with its structural pointer):\n\n{context}\n\n"
            f"Attached below are the actual images referenced above. Write the answer now."
        )
        answer_text = gemini.generate_multimodal(prompt, image_paths, system=SYNTH_SYSTEM)

        # Parse [SHOW: file | caption] directives -> resolved image list
        images_out = []
        clean_answer = answer_text
        for fname, caption in _SHOW_RE.findall(answer_text):
            fname = Path(fname.strip()).name
            path = filename_to_path.get(fname)
            if path and path.exists():
                images_out.append({"path": str(path), "caption": caption.strip(), "filename": fname})
        # Remove the raw directives from the visible answer text.
        clean_answer = _SHOW_RE.sub("", answer_text).strip()
        clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer)

        sources = [f["breadcrumb"] for f in finalists]
        return {
            "answer": clean_answer,
            "sources": sources,
            "images": images_out,
            "time_seconds": round(time.time() - t0, 2),
        }
