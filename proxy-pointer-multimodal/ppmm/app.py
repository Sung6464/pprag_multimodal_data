"""
Proxy-Pointer MultiModal — Streamlit UI.

Run:  streamlit run app.py

Features
  * Drop PDF / DOCX files into the UI -> they are extracted, structured into a
    proxy-pointer tree, embedded, and indexed on the fly (Gemini key only).
  * Ask questions -> grounded answers with the actual figures/tables rendered
    inline (visual citations), plus the structural pointers used as sources.
  * "Proxy Pointers" tab -> view every pointer stored in the ledger file as it
    is created, filter by document, and download the raw ledger.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pp import config, proxies
from pp.pipeline import list_indexed_docs, process_upload

st.set_page_config(page_title="Proxy-Pointer MultiModal", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_bot():
    # Imported lazily so the app still loads if the key is missing.
    from pp.rag_bot import MMRagBot

    return MMRagBot()


def _reset_bot():
    _get_bot.clear()


def _has_key() -> bool:
    return bool(config.GOOGLE_API_KEY)


# ---------------------------------------------------------------------------
# Sidebar — status, upload, indexed docs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Proxy-Pointer")
    st.caption("MultiModal RAG · structure-aware retrieval with visual citations")

    if _has_key():
        st.success("Gemini API key detected", icon="✅")
    else:
        st.error("No Gemini key. Add GOOGLE_API_KEY to a .env file.", icon="⚠️")

    st.divider()
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Drop PDF or DOCX files here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Files are extracted locally (PyMuPDF / python-docx) — no Adobe key needed.",
    )
    if uploads and st.button("Process & index", type="primary", use_container_width=True):
        if not _has_key():
            st.error("A Gemini key is required to embed & index documents.")
        else:
            prog = st.progress(0.0, text="Starting…")
            for i, up in enumerate(uploads):
                prog.progress(i / len(uploads), text=f"Processing {up.name}…")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(up.name).suffix
                ) as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = tmp.name
                try:
                    summary = process_upload(tmp_path, up.name)
                    st.write(
                        f"**{summary['doc_id']}** — {summary['n_indexed']} sections indexed, "
                        f"{summary['n_proxies']} proxy pointers created."
                    )
                except Exception as e:
                    st.error(f"Failed on {up.name}: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            prog.progress(1.0, text="Done")
            _reset_bot()
            st.toast("Indexing complete", icon="✅")

    st.divider()
    st.subheader("Indexed documents")
    docs = list_indexed_docs()
    if docs:
        for d in docs:
            n = len(proxies.load_proxies(d))
            st.write(f"• **{d}** — {n} pointers")
    else:
        st.caption("None yet. Upload a file above to get started.")


# ---------------------------------------------------------------------------
# Main — tabs
# ---------------------------------------------------------------------------
tab_ask, tab_proxies = st.tabs(["💬 Ask", "🧭 Proxy Pointers"])

# ---- Ask ------------------------------------------------------------------
with tab_ask:
    st.header("Ask your documents")
    st.caption(
        "Answers are grounded in full document sections and cite the actual "
        "figures/tables they rely on."
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. Describe the CLIP-CITE framework architecture and its loss components",
    )
    go = st.button("Ask", type="primary", disabled=not query.strip())

    if go:
        if not _has_key():
            st.error("Add a Gemini API key first.")
        elif not list_indexed_docs():
            st.warning("No documents indexed yet — upload a PDF or DOCX in the sidebar.")
        else:
            with st.spinner("Retrieving sections, re-ranking, and synthesizing…"):
                try:
                    result = _get_bot().answer(query.strip())
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    result = None

            if result:
                st.markdown(result["answer"])

                if result["images"]:
                    st.subheader("Visual evidence")
                    cols = st.columns(min(3, len(result["images"])))
                    for i, img in enumerate(result["images"]):
                        with cols[i % len(cols)]:
                            st.image(img["path"], caption=img["caption"], use_container_width=True)

                with st.expander(
                    f"Structural pointers used ({len(result['sources'])})", expanded=True
                ):
                    for s in result["sources"]:
                        st.markdown(f"- `{s}`")

                st.caption(f"Answered in {result['time_seconds']}s")

# ---- Proxy Pointers -------------------------------------------------------
with tab_proxies:
    st.header("Proxy-pointer ledger")
    st.caption(
        f"Every structural pointer is appended to `{config.PROXY_FILE}` as it is "
        "created. This is the separate file that stores all proxies."
    )

    s = proxies.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total pointers", s["total"])
    c2.metric("Documents", len(s["by_doc"]))
    c3.metric("Image anchors", s["total_image_anchors"])

    all_docs = sorted(s["by_doc"].keys())
    sel = st.selectbox("Filter by document", ["(all)"] + all_docs, index=0)
    rows = proxies.load_proxies(None if sel == "(all)" else sel)

    if not rows:
        st.info("No proxy pointers yet. Process a document to populate the ledger.")
    else:
        table = [
            {
                "created_at": r["created_at"],
                "doc": r["doc_id"],
                "node": r["node_id"],
                "level": r["level"],
                "proxy pointer (breadcrumb)": r["breadcrumb"],
                "images": len(r.get("images", [])),
                "chars": r.get("text_chars", 0),
            }
            for r in rows
        ]
        st.dataframe(table, use_container_width=True, hide_index=True, height=420)

        with st.expander("Inspect a pointer"):
            options = {f"{r['doc_id']} · {r['breadcrumb']}": r for r in rows}
            pick = st.selectbox("Pointer", list(options.keys()))
            r = options[pick]
            st.markdown(f"**Breadcrumb:** `{r['breadcrumb']}`")
            st.markdown(f"**Snippet:** {r['snippet']}")
            if r.get("images"):
                st.markdown("**Image anchors:**")
                for rel in r["images"]:
                    p = config.PAPERS_DIR / r["doc_id"] / rel
                    if p.exists():
                        st.image(str(p), caption=rel, width=280)
                    else:
                        st.caption(f"{rel} (missing)")

        if config.PROXY_FILE.exists():
            st.download_button(
                "⬇️ Download ledger (.jsonl)",
                data=config.PROXY_FILE.read_bytes(),
                file_name="proxy_pointers.jsonl",
                mime="application/json",
            )
