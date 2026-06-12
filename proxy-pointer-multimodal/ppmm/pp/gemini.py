"""Thin wrappers around the google-genai SDK (the current Gemini SDK)."""
from __future__ import annotations

import time
from pathlib import Path

from google import genai
from google.genai import types

from . import config

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_api_key())
    return _client


# ---------------------------------------------------------------------------
# Embeddings  (gemini-embedding-001, 1536-d to match the original)
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts. Batches to be polite to rate limits."""
    out: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        for attempt in range(5):
            try:
                resp = client().models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        output_dimensionality=config.EMBED_DIM,
                        task_type=task_type,
                    ),
                )
                out.extend([list(e.values) for e in resp.embeddings])
                break
            except Exception as e:  # transient rate limit / 5xx -> backoff
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


# ---------------------------------------------------------------------------
# Text generation (re-ranker)
# ---------------------------------------------------------------------------
def generate_text(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    cfg = types.GenerateContentConfig(temperature=temperature)
    if system:
        cfg.system_instruction = system
    resp = client().models.generate_content(
        model=config.GEN_MODEL, contents=prompt, config=cfg
    )
    return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# Multimodal generation (synthesizer): text + real images -> grounded answer
# ---------------------------------------------------------------------------
def generate_multimodal(
    prompt: str, image_paths: list[Path], system: str | None = None, temperature: float = 0.2
) -> str:
    parts: list = [prompt]
    for p in image_paths:
        p = Path(p)
        if not p.exists():
            continue
        ext = p.suffix.lower().lstrip(".")
        mime = "image/png" if ext not in ("jpg", "jpeg", "gif", "webp") else (
            "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        )
        try:
            parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
        except Exception:
            continue
    cfg = types.GenerateContentConfig(temperature=temperature)
    if system:
        cfg.system_instruction = system
    resp = client().models.generate_content(
        model=config.GEN_MODEL, contents=parts, config=cfg
    )
    return (resp.text or "").strip()
