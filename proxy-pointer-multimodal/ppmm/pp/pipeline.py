"""End-to-end orchestration: uploaded file -> markdown -> tree -> proxies -> index."""
from __future__ import annotations

from pathlib import Path

from . import config, extract, indexer, proxies
from .tree_builder import build_tree


def list_indexed_docs() -> list[str]:
    meta = indexer._load_meta()
    return sorted({m["doc_id"] for m in meta})


def process_upload(saved_path: str | Path, original_name: str) -> dict:
    """Run the full pipeline for one uploaded PDF/DOCX. Returns a summary."""
    # 1. Extract to markdown + images
    doc_id, out_dir = extract.extract_file(saved_path, original_name=original_name)

    # 2. If re-processing an existing doc, clear its old proxy pointers first.
    proxies.remove_doc(doc_id)

    # 3. Build structure tree (this appends fresh proxy pointers to the ledger).
    tree = build_tree(doc_id, out_dir / f"{doc_id}.md")

    # 4. Embed + index
    n_indexed = indexer.index_document(doc_id)

    new_proxies = proxies.load_proxies(doc_id)
    return {
        "doc_id": doc_id,
        "n_nodes": tree["n_nodes"],
        "n_indexed": n_indexed,
        "n_proxies": len(new_proxies),
        "md_path": str(out_dir / f"{doc_id}.md"),
    }
