"""
Index builder.

Embeds every structure-tree node (its breadcrumb + body) with Gemini and
stores them in a FAISS cosine index. Metadata for each vector keeps the
breadcrumb (proxy pointer), full text, snippet, and image anchors so the
retriever can load the full multimodal payload later.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from . import config, gemini
from .tree_builder import iter_nodes, load_tree


def _node_embedding_text(node: dict) -> str:
    # Embed the structural pointer together with the body for "structure-aware"
    # recall. Cap body length to keep embedding calls cheap.
    body = node.get("text", "")[:4000]
    return f"{node['breadcrumb']}\n\n{body}".strip()


def _load_meta() -> list[dict]:
    if config.META_PATH.exists():
        return json.loads(config.META_PATH.read_text(encoding="utf-8"))
    return []


def _save_meta(meta: list[dict]):
    config.META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _load_index() -> faiss.Index | None:
    if config.INDEX_PATH.exists():
        return faiss.read_index(str(config.INDEX_PATH))
    return None


def _new_index() -> faiss.Index:
    # Inner product on L2-normalized vectors == cosine similarity.
    return faiss.IndexFlatIP(config.EMBED_DIM)


def index_document(doc_id: str) -> int:
    """(Re)index one document. Returns number of nodes indexed."""
    tree = load_tree(doc_id)
    nodes = [n for n in iter_nodes(tree) if n.get("text") or n.get("images")]
    if not nodes:
        return 0

    texts = [_node_embedding_text(n) for n in nodes]
    vecs = np.array(gemini.embed_texts(texts), dtype="float32")
    faiss.normalize_L2(vecs)

    # Drop any previous vectors for this doc, then append the new ones.
    meta = [m for m in _load_meta() if m["doc_id"] != doc_id]
    index = _new_index()
    if meta:
        # rebuild from scratch is simplest & safe for FlatIP; re-embed avoided
        # by storing vectors in meta. To keep meta light we instead rebuild only
        # from current docs' stored vectors.
        old_vecs = np.array([m["_vec"] for m in meta], dtype="float32")
        index.add(old_vecs)

    new_meta = []
    for n in nodes:
        new_meta.append(
            {
                "doc_id": doc_id,
                "node_id": n["node_id"],
                "breadcrumb": n["breadcrumb"],
                "title": n["title"],
                "snippet": n["snippet"],
                "text": n["text"],
                "images": n["images"],
            }
        )
    index.add(vecs)
    # attach raw vectors to meta so we can rebuild without re-embedding
    for m, v in zip(new_meta, vecs.tolist()):
        m["_vec"] = v

    meta = meta + new_meta
    faiss.write_index(index, str(config.INDEX_PATH))
    _save_meta(meta)
    return len(nodes)


class RetrievalStore:
    """Loaded FAISS index + metadata, ready for search."""

    def __init__(self):
        self.index = _load_index()
        self.meta = _load_meta()

    @property
    def ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0 and bool(self.meta)

    def search(self, qvec: np.ndarray, k: int):
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m = self.meta[i]
            results.append({**{kk: vv for kk, vv in m.items() if kk != "_vec"}, "score": float(score)})
        return results
