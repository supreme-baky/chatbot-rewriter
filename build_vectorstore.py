# build_vectorstore.py

from functions import load_documents, create_vectorstore

if __name__ == "__main__":
    docs = load_documents("./data")  # put your Strunk & White PDF inside ./docs/
    create_vectorstore(docs, persist_dir="./chroma_db")
