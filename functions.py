import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Updated vector store & embedding imports (no deprecation warnings)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Updated loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)

# Document class
from langchain_core.documents import Document

# Optional Whisper (fail-safe import)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# ----------------------------
# DOCUMENT LOADING
# ----------------------------
def load_documents(path: str) -> List[Document]:
    """Load PDF, TXT, CSV, and MP3 files as LangChain Documents."""
    docs = []

    for file in os.listdir(path):
        full_path = os.path.join(path, file)

        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(full_path).load())

        elif file.endswith(".txt"):
            docs.extend(TextLoader(full_path).load())

        elif file.endswith(".csv"):
            docs.extend(CSVLoader(full_path).load())

        elif file.endswith(".mp3"):
            print(f"🎙️ Transcribing audio: {file}")
            docs.extend(transcribe_audio(full_path))

    print(f"✅ Loaded {len(docs)} documents from {path}")
    return docs


# ----------------------------
# AUDIO TRANSCRIPTION (SAFE MODE)
# ----------------------------
def transcribe_audio(file_path: str) -> List[Document]:
    """Convert MP3 into text using faster-whisper, fallback to empty."""
    if not WHISPER_AVAILABLE:
        print("⚠️ Whisper not installed. Skipping audio transcription.")
        return []

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model = WhisperModel("base", device=device, compute_type="int8")

    segments, _ = model.transcribe(file_path)
    text = " ".join([segment.text for segment in segments]).strip()

    return [Document(page_content=text, metadata={"source": file_path})]


# ----------------------------
# VECTOR STORE OPERATIONS
# ----------------------------
def create_vectorstore(docs: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    """Split, embed, and persist documents as a Chroma vector DB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ Vectorstore persisted at {persist_dir}")
    return vectordb


def load_vectorstore(persist_dir: str = "./chroma_db") -> Chroma:
    """Load existing Chroma store or raise an error."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"❌ No vectorstore found at {persist_dir}. Run create_vectorstore() first."
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
