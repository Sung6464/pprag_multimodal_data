"""
Structure-tree builder.

Parses a document's markdown into a hierarchy of nodes (Section > Sub-section).
Each node carries:
    * a breadcrumb  ("CLIP > 3 Method > 3.1 ...")  <-- the PROXY POINTER
    * the full body text of that section
    * the image anchors (figures/tables) physically located inside it

The breadcrumb IS the "proxy pointer" the architecture indexes instead of raw
chunks. We persist every pointer to a separate append-only file so they can be
viewed in the UI as they are created.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .proxies import record_proxies

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _slug(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_tree(doc_id: str, md_path: Path) -> dict:
    """Build the structure tree and return it as a dict. Also writes
    trees/<doc_id>.tree.json and appends proxy pointers to the proxy file."""
    text = Path(md_path).read_text(encoding="utf-8")
    lines = text.split("\n")

    root = {
        "node_id": "n0",
        "doc_id": doc_id,
        "title": doc_id,
        "level": 0,
        "breadcrumb": doc_id,
        "body_lines": [],
        "images": [],
        "children": [],
    }
    counter = [0]

    def new_node(title: str, level: int, breadcrumb: str) -> dict:
        counter[0] += 1
        return {
            "node_id": f"n{counter[0]}",
            "doc_id": doc_id,
            "title": title,
            "level": level,
            "breadcrumb": breadcrumb,
            "body_lines": [],
            "images": [],
            "children": [],
        }

    # stack holds (level, node); root is level 0
    stack: list[tuple[int, dict]] = [(0, root)]

    for raw in lines:
        m = _HEADING.match(raw)
        if m:
            level = len(m.group(1))
            title = _slug(m.group(2))
            # pop until parent has a smaller level
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                stack = [(0, root)]
            parent = stack[-1][1]
            breadcrumb = f"{parent['breadcrumb']} > {title}"
            node = new_node(title, level, breadcrumb)
            parent["children"].append(node)
            stack.append((level, node))
        else:
            cur = stack[-1][1]
            cur["body_lines"].append(raw)
            for im in _IMG.findall(raw):
                cur["images"].append(im.strip())

    # finalize: collapse body_lines -> body text + snippet
    proxies: list[dict] = []

    def finalize(node: dict):
        body = "\n".join(node["body_lines"]).strip()
        node["text"] = body
        node["snippet"] = re.sub(r"\s+", " ", body)[: config.SNIPPET_CHARS].strip()
        node.pop("body_lines", None)
        # Every node with content OR images becomes a retrievable proxy pointer.
        if node["level"] > 0 and (body or node["images"]):
            proxies.append(
                {
                    "doc_id": doc_id,
                    "node_id": node["node_id"],
                    "breadcrumb": node["breadcrumb"],
                    "level": node["level"],
                    "snippet": node["snippet"],
                    "images": list(node["images"]),
                    "text_chars": len(body),
                }
            )
        for ch in node["children"]:
            finalize(ch)

    finalize(root)

    tree = {"doc_id": doc_id, "root": root, "n_nodes": counter[0]}
    out_path = config.TREES_DIR / f"{doc_id}.tree.json"
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

    # Persist + return the pointers (UI reads them from the proxy file).
    record_proxies(proxies)
    return tree


def iter_nodes(tree: dict):
    """Yield every content node (skips the synthetic root)."""
    def walk(node):
        if node["level"] > 0:
            yield node
        for ch in node.get("children", []):
            yield from walk(ch)
    yield from walk(tree["root"])


def load_tree(doc_id: str) -> dict:
    return json.loads((config.TREES_DIR / f"{doc_id}.tree.json").read_text(encoding="utf-8"))
