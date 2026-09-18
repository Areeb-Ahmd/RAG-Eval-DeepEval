import os
import re
import glob
import time

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads GOOGLE_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if os.path.exists(DB_DIR):
        store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        if store._collection.count() > 0:
            return store

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    print(f"Embedding {len(chunks)} chunks into Chroma in batches...")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    batch_size = 30
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for attempt in range(5):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("⚠️ Quota limit hit (429). Waiting 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    raise e
        time.sleep(2)  # Pause between batches to prevent rate limits

    return vector_store


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")