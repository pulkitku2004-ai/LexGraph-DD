"""
Sprint 10 — Streamlit UI for the Legal Due Diligence Engine.

Three states drive the entire UI:
  1. UPLOAD  — no active job; show file uploader + submit button
  2. RUNNING — job submitted; poll API every 5s, show progress spinner
  3. DONE    — job complete; show Report tab + Q&A tab

Session state keys:
  job_id      str | None  — active job UUID
  job_status  str         — "pending" | "running" | "done" | "error"
  report      str | None  — final markdown report
  doc_ids     list[str]   — document identifiers for the active job
  errors      list[str]   — pipeline errors from the job

Start the API first, then run this app:
  uvicorn legal_due_diligence.api.main:app --port 8000
  streamlit run legal_due_diligence/ui/app.py
"""

from __future__ import annotations

import time

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Due Diligence",
    page_icon="⚖️",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "job_id": None,
    "job_status": None,
    "report": None,
    "doc_ids": [],
    "errors": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal DD Engine")
    api_url = st.text_input("API URL", value="http://localhost:8000", key="api_url")

    if st.session_state.job_id:
        st.divider()
        st.markdown(f"**Active job**")
        st.code(st.session_state.job_id, language=None)
        st.markdown(f"Status: `{st.session_state.job_status}`")
        if st.session_state.doc_ids:
            st.markdown("**Documents:**")
            for d in st.session_state.doc_ids:
                st.markdown(f"- {d}")

        st.divider()
        if st.button("🗑️ Delete job & clear data", type="secondary", use_container_width=True):
            try:
                r = requests.delete(f"{api_url}/jobs/{st.session_state.job_id}", timeout=10)
                if r.status_code in (204, 404):
                    for key in ("job_id", "job_status", "report", "doc_ids", "errors"):
                        st.session_state[key] = None if key in ("job_id", "job_status", "report") else []
                    st.success("Job deleted.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Delete failed: {r.status_code} {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is the server running?")


# ── Helper: poll job status ────────────────────────────────────────────────────
def _poll(job_id: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/jobs/{job_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.ConnectionError:
        pass
    return None


# ── State: UPLOAD ─────────────────────────────────────────────────────────────
if st.session_state.job_id is None:
    st.title("Legal Due Diligence")
    st.markdown("Upload contract files to run the full analysis pipeline: clause extraction, risk scoring, contradiction detection, and Q&A.")

    uploaded = st.file_uploader(
        "Upload contracts (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) selected:**")
        for f in uploaded:
            st.markdown(f"- {f.name} ({f.size / 1024:.1f} KB)")

    col1, col2 = st.columns([2, 5])
    with col1:
        submit = st.button(
            "▶ Run Analysis",
            type="primary",
            disabled=not uploaded,
            use_container_width=True,
        )

    if submit and uploaded:
        files = [("files", (f.name, f.read(), f.type or "application/octet-stream")) for f in uploaded]
        try:
            with st.spinner("Submitting job…"):
                r = requests.post(f"{api_url}/jobs", files=files, timeout=30)
            if r.status_code == 202:
                data = r.json()
                st.session_state.job_id = data["job_id"]
                st.session_state.job_status = data["status"]
                st.session_state.doc_ids = data["doc_ids"]
                st.session_state.errors = data.get("errors", [])
                st.rerun()
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Start it with:\n\n```\nuvicorn legal_due_diligence.api.main:app --port 8000\n```")


# ── State: RUNNING ─────────────────────────────────────────────────────────────
elif st.session_state.job_status in ("pending", "running"):
    st.title("Analysing contracts…")
    st.markdown(f"Job `{st.session_state.job_id}` is running. This takes a few minutes for LLM extraction across all CUAD categories.")

    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.progress(0.0, text="Waiting for pipeline to start…")

    with st.spinner("Running clause extraction, risk scoring, contradiction detection…"):
        while True:
            data = _poll(st.session_state.job_id)
            if data is None:
                st.error("Lost connection to API.")
                break

            st.session_state.job_status = data["status"]
            st.session_state.errors = data.get("errors", [])

            if data["status"] == "done":
                st.session_state.report = data.get("report")
                progress_placeholder.progress(1.0, text="Complete!")
                time.sleep(0.3)
                st.rerun()
                break
            elif data["status"] == "error":
                progress_placeholder.empty()
                st.rerun()
                break

            time.sleep(5)


# ── State: ERROR ───────────────────────────────────────────────────────────────
elif st.session_state.job_status == "error":
    st.title("Pipeline error")
    st.error("The pipeline encountered an error and could not complete.")
    if st.session_state.errors:
        with st.expander("Error details"):
            for e in st.session_state.errors:
                st.markdown(f"- {e}")
    st.markdown("Delete this job from the sidebar and try again.")


# ── State: DONE ────────────────────────────────────────────────────────────────
elif st.session_state.job_status == "done":
    st.title("Due Diligence Report")

    if st.session_state.errors:
        with st.expander(f"⚠️ {len(st.session_state.errors)} warning(s) during processing"):
            for e in st.session_state.errors:
                st.markdown(f"- {e}")

    tab_report, tab_qa = st.tabs(["📄 Report", "💬 Q&A"])

    # ── Report tab ─────────────────────────────────────────────────────────
    with tab_report:
        if st.session_state.report:
            st.markdown(st.session_state.report)
        else:
            st.warning("Report is empty — the pipeline may have encountered extraction issues.")

    # ── Q&A tab ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown("Ask a question grounded in the uploaded contracts. Answers include page-level citations.")

        question = st.text_input(
            "Question",
            placeholder="What is the governing law for each contract?",
            key="qa_input",
        )

        ask = st.button("Ask", type="primary", disabled=not question.strip())

        if ask and question.strip():
            try:
                with st.spinner("Retrieving relevant clauses and generating answer…"):
                    r = requests.post(
                        f"{api_url}/jobs/{st.session_state.job_id}/qa",
                        json={"question": question},
                        timeout=60,
                    )
                if r.status_code == 200:
                    result = r.json()
                    st.markdown("### Answer")
                    st.markdown(result["answer"])

                    if result["citations"]:
                        st.markdown("### Citations")
                        for i, c in enumerate(result["citations"], 1):
                            with st.expander(f"[{i}] {c['doc_id']} — page {c['page_number']}"):
                                st.markdown(f"**Chunk ID:** `{c['chunk_id']}`")
                                st.markdown(f"> {c['excerpt']}")
                else:
                    st.error(f"Q&A error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API.")
