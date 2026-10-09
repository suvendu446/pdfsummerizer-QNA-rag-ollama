"""Core RAG logic: ingestion, indexing, retrieval and generation with local Ollama models.

Pipeline
    PDFs -> pages -> chunks -> nomic-embed-text embeddings -> FAISS index   (build_index)
    question -> (rewrite using chat history) -> top-k retrieval
             -> grounded prompt -> local LLM via Ollama -> answer + citations   (answer_question)
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL_ID = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL_ID = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
QA_SYSTEM = (
    "You are a careful assistant that answers questions about research documents. "
    "Use ONLY the provided context. If the answer is not in the context, say that you "
    "don't know instead of guessing. Cite the sources you used inline as [1], [2], ... "
    "matching the numbered context blocks. Be clear and reasonably concise."
)

QA_PROMPT = """Context:
{context}

Question: {question}

Answer (with inline citations like [1]):"""

CONDENSE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up \
as a standalone question that can be understood without the conversation. If it is already \
standalone, return it unchanged. Return only the question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

SUMMARY_PROMPT = """Write a structured summary of the research material below with these \
headings: Objective, Methods, Key Findings, Limitations. Use only the material provided; if a \
heading is not covered, write "Not stated". Keep it under 300 words.

Material:
{text}"""

PARTIAL_SUMMARY_PROMPT = """Summarize this part of a research paper in 5-8 bullet points. \
Capture objectives, methods, data, results and limitations if present. Use only the text.

Text:
{text}"""


# ----------------------------------------------------------------------------
# Ollama (local models, free, no limits)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_ollama():
    import ollama

    return ollama.Client(host=OLLAMA_HOST)


class OllamaEmbeddings(Embeddings):
    """Minimal LangChain-compatible embeddings wrapper around a local Ollama model."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), 32):  # small batches keep memory use low
            response = get_ollama().embed(model=self.model, input=texts[start:start + 32])
            vectors.extend(response["embeddings"])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    return OllamaEmbeddings(EMBED_MODEL_ID)


def generate(prompt, system=None, max_tokens=700, temperature=0.2, model_id=None) -> str:
    """One chat call to the local model."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    response = get_ollama().chat(
        model=model_id or LLM_MODEL_ID,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
    )
    return response["message"]["content"].strip()


# ----------------------------------------------------------------------------
# Ingestion + indexing
# ----------------------------------------------------------------------------
def list_pdfs(data_dir: Path = DATA_DIR):
    data_dir.mkdir(exist_ok=True)
    return sorted(data_dir.glob("*.pdf"))


def load_pdfs(data_dir: Path = DATA_DIR):
    """Load every PDF page as a Document. Returns (documents, skipped_page_count).

    Pages without extractable text (e.g. scanned images) are skipped and counted.
    """
    documents, skipped = [], 0
    for pdf in list_pdfs(data_dir):
        for page in PyPDFLoader(str(pdf)).load():
            if not page.page_content.strip():
                skipped += 1
                continue
            page.metadata["source"] = pdf.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1  # 1-based
            documents.append(page)
    return documents, skipped


def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_index(chunk_size=1000, chunk_overlap=150, index_dir: Path = INDEX_DIR,
                data_dir: Path = DATA_DIR) -> dict:
    """Read PDFs, chunk them, embed the chunks and save a FAISS index to disk."""
    documents, skipped = load_pdfs(data_dir)
    if not documents:
        raise ValueError("No extractable text found. Add text-based PDFs to the data/ folder.")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(index_dir))
    return {
        "pdfs": len(list_pdfs(data_dir)),
        "pages": len(documents),
        "chunks": len(chunks),
        "skipped_pages": skipped,
    }


def load_index(index_dir: Path = INDEX_DIR) -> FAISS:
    # allow_dangerous_deserialization is required by LangChain because the index uses
    # pickle. It is safe here because we only load an index we created ourselves.
    return FAISS.load_local(
        str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
    )


def index_exists(index_dir: Path = INDEX_DIR) -> bool:
    return (index_dir / "index.faiss").exists()


# ----------------------------------------------------------------------------
# Retrieval + answering
# ----------------------------------------------------------------------------
def retrieve(store: FAISS, query: str, k: int = 4):
    """Top-k similar chunks as (Document, distance) pairs. Lower distance = closer."""
    return store.similarity_search_with_score(query, k=k)


def _format_history(history, max_turns=6, max_chars=500):
    lines = []
    for msg in history[-max_turns:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:max_chars]}")
    return "\n".join(lines)


def condense_question(history, question, model_id=None) -> str:
    """Rewrite a follow-up ("explain that simply") into a standalone search query."""
    prompt = CONDENSE_PROMPT.format(history=_format_history(history), question=question)
    rewritten = generate(prompt, max_tokens=150, temperature=0.0, model_id=model_id)
    return rewritten or question


def answer_question(store: FAISS, question: str, history=None, k: int = 4,
                    model_id=None) -> dict:
    standalone = condense_question(history, question, model_id) if history else question
    results = retrieve(store, standalone, k)

    context = "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source')}, page {doc.metadata.get('page')})\n"
        f"{doc.page_content}"
        for i, (doc, _) in enumerate(results, start=1)
    )
    text = generate(
        QA_PROMPT.format(context=context, question=standalone),
        system=QA_SYSTEM,
        model_id=model_id,
    )
    sources = [
        {
            "id": i,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:400],
            "distance": float(score),
        }
        for i, (doc, score) in enumerate(results, start=1)
    ]
    return {"answer": text, "standalone_question": standalone, "sources": sources}


# ----------------------------------------------------------------------------
# Summarization (map-reduce over a single paper)
# ----------------------------------------------------------------------------
def summarize_pdf(pdf_path: Path, model_id=None, block_chars: int = 8000) -> str:
    pages = PyPDFLoader(str(pdf_path)).load()
    text = "\n\n".join(
        f"[Page {int(p.metadata.get('page', 0)) + 1}]\n{p.page_content}"
        for p in pages
        if p.page_content.strip()
    )
    if not text.strip():
        raise ValueError("No extractable text in this PDF (it may be a scanned document).")

    blocks = RecursiveCharacterTextSplitter(
        chunk_size=block_chars, chunk_overlap=200
    ).split_text(text)

    if len(blocks) == 1:
        return generate(SUMMARY_PROMPT.format(text=blocks[0]), max_tokens=800, model_id=model_id)

    # map: summarize each block, reduce: combine the partial summaries
    partials = [
        generate(PARTIAL_SUMMARY_PROMPT.format(text=b), max_tokens=500, model_id=model_id)
        for b in blocks
    ]
    return generate(
        SUMMARY_PROMPT.format(text="\n\n".join(partials)), max_tokens=800, model_id=model_id
    )
