import streamlit as st

import rag

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("📄 Research Paper Assistant")
st.caption("RAG with page citations and follow-up questions. Runs fully on your PC with Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_error(exc: Exception):
    text = str(exc)
    if "connect" in text.lower() or "refused" in text.lower():
        st.error("Cannot reach Ollama. Start the Ollama app (or run `ollama serve`) and try again.")
    elif "not found" in text.lower():
        st.error(f"Ollama model missing. Run: ollama pull {rag.LLM_MODEL_ID} and "
                 f"ollama pull {rag.EMBED_MODEL_ID}")
    else:
        st.error(text)


@st.cache_resource
def get_store(index_mtime: float):
    """Load the FAISS index once; reload automatically when the index file changes."""
    return rag.load_index()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads:
        rag.DATA_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (rag.DATA_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to data/")

    pdfs = rag.list_pdfs()
    st.write(f"PDFs in data/: **{len(pdfs)}**")

    st.header("2. Index")
    chunk_size = st.number_input("Chunk size (characters)", 200, 4000, 1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 150, step=50)
    if st.button("Build / rebuild index", disabled=not pdfs):
        with st.spinner("Reading PDFs and creating embeddings..."):
            try:
                stats = rag.build_index(int(chunk_size), int(chunk_overlap))
                st.success(
                    f"Indexed {stats['pdfs']} PDF(s), {stats['pages']} pages, "
                    f"{stats['chunks']} chunks."
                )
                if stats["skipped_pages"]:
                    st.warning(f"{stats['skipped_pages']} page(s) had no extractable text (scanned?).")
            except Exception as exc:  # noqa: BLE001
                show_error(exc)

    st.header("3. Settings")
    model_id = st.text_input("Chat model", value=rag.LLM_MODEL_ID)
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, 4)
    st.caption(f"Embeddings: {rag.EMBED_MODEL_ID}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------- main
chat_tab, summary_tab = st.tabs(["💬 Ask questions", "📝 Summarize a paper"])

with chat_tab:
    if not rag.index_exists():
        st.info("Upload PDFs and click **Build / rebuild index** in the sidebar to start.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"**[{s['id']}] {s['source']} - page {s['page']}**")
                            st.caption(s["snippet"] + "...")

        question = st.chat_input("Ask a question about your papers")
        if question:
            history = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        store = get_store(rag.INDEX_DIR.joinpath("index.faiss").stat().st_mtime)
                        result = rag.answer_question(
                            store, question, history=history, k=top_k, model_id=model_id
                        )
                        st.markdown(result["answer"])
                        if result["standalone_question"] != question:
                            st.caption(f"Searched for: {result['standalone_question']}")
                        with st.expander("Sources"):
                            for s in result["sources"]:
                                st.markdown(f"**[{s['id']}] {s['source']} - page {s['page']}**")
                                st.caption(s["snippet"] + "...")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["answer"],
                             "sources": result["sources"]}
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.messages.pop()  # drop the unanswered question
                        show_error(exc)

with summary_tab:
    if not pdfs:
        st.info("Upload a PDF in the sidebar first.")
    else:
        choice = st.selectbox("Choose a paper", [p.name for p in pdfs])
        if st.button("Summarize"):
            with st.spinner("Summarizing (long papers take a few calls)..."):
                try:
                    st.markdown(rag.summarize_pdf(rag.DATA_DIR / choice, model_id=model_id))
                except Exception as exc:  # noqa: BLE001
                    show_error(exc)
