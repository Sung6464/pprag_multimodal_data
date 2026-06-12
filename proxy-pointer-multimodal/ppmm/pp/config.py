"""Central configuration. Override any value via environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
# Accept either GOOGLE_API_KEY or GEMINI_API_KEY so the app is forgiving.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

# ---------------------------------------------------------------------------
# Models  (verified current Gemini model strings, June 2026)
# ---------------------------------------------------------------------------
# gemini-3.1-flash-lite is multimodal (text+image in) and cheap -> used for
# BOTH the re-ranker and the multimodal synthesizer (its vision does the
# "visual grounding"). Override with GEN_MODEL if you prefer e.g. gemini-3-flash.
GEN_MODEL = os.getenv("PP_GEN_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("PP_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("PP_EMBED_DIM", "1536"))

# ---------------------------------------------------------------------------
# Retrieval knobs (mirror the original Proxy-Pointer MultiModal pipeline)
# ---------------------------------------------------------------------------
RECALL_K = int(os.getenv("PP_RECALL_K", "200"))      # broad vector recall
CANDIDATE_K = int(os.getenv("PP_CANDIDATE_K", "50"))  # unique candidates to re-rank
FINALIST_K = int(os.getenv("PP_FINALIST_K", "5"))     # nodes loaded for synthesis
SNIPPET_CHARS = int(os.getenv("PP_SNIPPET_CHARS", "150"))
MAX_IMAGES_PER_ANSWER = int(os.getenv("PP_MAX_IMAGES", "6"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent  # project root (ppmm/)
DATA_DIR = Path(os.getenv("PP_DATA_DIR", _DEFAULT_ROOT / "data"))

PAPERS_DIR = Path(os.getenv("PP_PAPERS_DIR", DATA_DIR / "extracted_papers"))
TREES_DIR = Path(os.getenv("PP_TREES_DIR", DATA_DIR / "trees"))
INDEX_DIR = Path(os.getenv("PP_INDEX_DIR", DATA_DIR / "index"))
PROXIES_DIR = Path(os.getenv("PP_PROXIES_DIR", DATA_DIR / "proxies"))

# The "diff file" that stores every proxy-pointer as it is created.
PROXY_FILE = Path(os.getenv("PP_PROXY_FILE", PROXIES_DIR / "proxy_pointers.jsonl"))

INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

for _d in (PAPERS_DIR, TREES_DIR, INDEX_DIR, PROXIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "No Gemini API key found. Put GOOGLE_API_KEY=... (or GEMINI_API_KEY=...) "
            "in a .env file next to app.py, or set it in your environment."
        )
    return GOOGLE_API_KEY
