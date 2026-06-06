import logging

import streamlit as st

from core.app_utils import get_client, RETRO_STYLING
from core.rag_engine import LegalRAG, get_judgment_hash
from config import PAGE_HOME

# Apply the same styling as other pages
st.markdown(RETRO_STYLING, unsafe_allow_html=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db():
    """Return a short-lived DB session (caller must close it)."""
    from database import SessionLocal
    return SessionLocal()


def _load_case_document_text(case_id: int) -> str | None:
    """Fetch the latest document text for *case_id* from the database."""
    try:
        from db.crud.knowledge import get_latest_case_document_text
        db = _get_db()
        try:
            return get_latest_case_document_text(db, case_id=case_id)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not load case document text for case %s: %s", case_id, exc)
        return None


def _case_has_pending_invalidations(case_id: int) -> bool:
    """Return True when the backend has unprocessed invalidation records for *case_id*."""
    try:
        from db.crud.knowledge import has_pending_invalidations
        db = _get_db()
        try:
            return has_pending_invalidations(db, case_id=case_id)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not check invalidation status for case %s: %s", case_id, exc)
        return False


# ---------------------------------------------------------------------------
# RAG engine cache – keyed by document hash so a new hash forces a new engine
# ---------------------------------------------------------------------------

@st.cache_resource
def get_rag_engine(text_hash: str):  # noqa: ARG001 – hash is the cache key
    return LegalRAG()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_page():
    st.title("💬 Chat with Judgment")

    # ------------------------------------------------------------------
    # 1. Resolve document text
    #    Priority: case_id from session → session judgment_raw_text
    # ------------------------------------------------------------------
    case_id: int | None = st.session_state.get("active_case_id")
    raw_text: str | None = None

    if case_id is not None:
        # Try to load the freshest document text from the DB for this case
        db_text = _load_case_document_text(case_id)
        if db_text:
            raw_text = db_text
            # Keep session state in sync so other pages see the same text
            st.session_state.judgment_raw_text = raw_text
        else:
            # Fall back to whatever was uploaded in this session
            raw_text = st.session_state.get("judgment_raw_text")
    else:
        raw_text = st.session_state.get("judgment_raw_text")

    if not raw_text:
        st.warning(
            "No judgment document found. Please go back to the Home page and upload a document first."
        )
        if st.button("⬅️ Back to Home"):
            st.switch_page(PAGE_HOME)
        return

    # ------------------------------------------------------------------
    # 2. Detect stale knowledge
    #    a) Hash-based: document text changed since last RAG init
    #    b) Invalidation-based: backend has pending recompute records
    # ------------------------------------------------------------------
    current_hash = get_judgment_hash(raw_text)
    hash_changed = st.session_state.get("last_judgment_hash") != current_hash

    backend_stale = False
    if case_id is not None:
        backend_stale = _case_has_pending_invalidations(case_id)

    knowledge_is_stale = hash_changed or backend_stale

    if knowledge_is_stale:
        # Reset chat and RAG so the user gets fresh answers
        st.session_state.chat_history = []
        st.session_state.rag_initialized = False
        st.session_state.last_judgment_hash = current_hash

        if backend_stale and not hash_changed:
            # The document text itself hasn't changed in this session, but the
            # backend has recomputed embeddings – reload the latest text.
            if case_id is not None:
                refreshed = _load_case_document_text(case_id)
                if refreshed and refreshed != raw_text:
                    raw_text = refreshed
                    st.session_state.judgment_raw_text = raw_text
                    current_hash = get_judgment_hash(raw_text)
                    st.session_state.last_judgment_hash = current_hash

    # ------------------------------------------------------------------
    # 3. Knowledge-status banner
    # ------------------------------------------------------------------
    if backend_stale:
        st.warning(
            "⚠️ **Knowledge is being updated** – the backend is recomputing embeddings "
            "for this case. Chat has been reset to avoid stale answers. "
            "Refresh the page once the update completes for the freshest context.",
            icon="⚠️",
        )
    elif hash_changed and not backend_stale:
        st.info("ℹ️ Document changed – chat context has been refreshed.", icon="ℹ️")

    # ------------------------------------------------------------------
    # 4. Sidebar controls
    # ------------------------------------------------------------------
    language = st.session_state.get("judgment_language", "English")
    st.sidebar.markdown(f"**Chat Language:** {language}")
    st.sidebar.info("You can change the language on the Home page.")

    if case_id is not None:
        st.sidebar.markdown(f"**Case ID:** `{case_id}`")

    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.rag_initialized = False
        stale_hash = st.session_state.get("last_judgment_hash", "")
        rag_engine = get_rag_engine(stale_hash)
        rag_engine.reset()
        st.rerun()

    if case_id is not None and st.sidebar.button("🔄 Refresh from Case", use_container_width=True):
        refreshed = _load_case_document_text(case_id)
        if refreshed:
            st.session_state.judgment_raw_text = refreshed
            st.session_state.rag_initialized = False
            st.session_state.chat_history = []
            st.session_state.last_judgment_hash = get_judgment_hash(refreshed)
        st.rerun()

    # ------------------------------------------------------------------
    # 5. Initialize RAG engine
    # ------------------------------------------------------------------
    rag_engine = get_rag_engine(current_hash)

    if not st.session_state.get("rag_initialized"):
        with st.spinner(
            "Initializing interactive chat… (this may take a moment to process the document)"
        ):
            success = rag_engine.initialize_vector_store(raw_text)
            if success:
                st.session_state.rag_initialized = True
            else:
                st.error("Failed to initialize chat engine.")
                return

    # ------------------------------------------------------------------
    # 6. Chat UI
    # ------------------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown(
        "Ask any specific questions about the uploaded judgment. "
        "The AI will answer strictly based on the document's contents."
    )
    st.markdown("---")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input (text or audio)
    audio_val = st.audio_input("🎤 Or speak your question...")

    user_question = None

    if prompt := st.chat_input("Ask a question about the judgment..."):
        user_question = prompt
    elif audio_val is not None:
        audio_id = hash(audio_val.getvalue())
        if st.session_state.get("last_processed_audio_id") != audio_id:
            with st.spinner("Transcribing audio..."):
                from core.audio_utils import transcribe_audio
                transcribed_text = transcribe_audio(audio_val.getvalue())
                if transcribed_text:
                    user_question = transcribed_text
                    st.session_state["last_processed_audio_id"] = audio_id
                else:
                    st.error("Failed to transcribe audio.")

    if user_question:
        # Warn if knowledge is still stale (backend hasn't finished recomputing)
        if backend_stale:
            st.warning(
                "⚠️ Answering with potentially stale knowledge – "
                "backend recompute is still in progress.",
                icon="⚠️",
            )

        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_client()
                    response = rag_engine.query(
                        question=user_question,
                        language=language,
                        openai_client=client,
                        chat_history=st.session_state.chat_history,
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error communicating with AI: {str(e)}")


if __name__ == "__main__":
    render_page()
