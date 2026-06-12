"""
Proxy-pointer ledger.

Every structural pointer (breadcrumb) created while indexing a document is
appended to a single human-readable file:  data/proxies/proxy_pointers.jsonl

Each line is one JSON record:
    {
      "created_at": "2026-06-12T10:31:02Z",
      "doc_id": "CLIP",
      "node_id": "n12",
      "breadcrumb": "CLIP > 3 Method > 3.1 Discriminative ...",
      "level": 2,
      "snippet": "first 150 chars of the section ...",
      "images": ["figures/img_4.png"],
      "text_chars": 812
    }

The Streamlit UI reads this file to show every proxy as it is created.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_proxies(proxies: list[dict]) -> int:
    """Append a batch of proxy pointers to the ledger. Returns count written."""
    config.PROXY_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = _now()
    with config.PROXY_FILE.open("a", encoding="utf-8") as f:
        for p in proxies:
            rec = {"created_at": ts, **p}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(proxies)


def load_proxies(doc_id: str | None = None) -> list[dict]:
    """Read all proxy pointers (optionally filtered by doc_id), newest first."""
    if not config.PROXY_FILE.exists():
        return []
    rows: list[dict] = []
    for line in config.PROXY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if doc_id and rec.get("doc_id") != doc_id:
            continue
        rows.append(rec)
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows


def remove_doc(doc_id: str) -> int:
    """Drop all pointers for a doc (used when re-indexing). Returns kept count."""
    if not config.PROXY_FILE.exists():
        return 0
    kept = [
        l
        for l in config.PROXY_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("doc_id") != doc_id
    ]
    config.PROXY_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def clear_all() -> None:
    if config.PROXY_FILE.exists():
        config.PROXY_FILE.unlink()


def stats() -> dict:
    rows = load_proxies()
    by_doc: dict[str, int] = {}
    imgs = 0
    for r in rows:
        by_doc[r["doc_id"]] = by_doc.get(r["doc_id"], 0) + 1
        imgs += len(r.get("images", []))
    return {"total": len(rows), "by_doc": by_doc, "total_image_anchors": imgs}
