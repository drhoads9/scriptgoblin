import os
import faiss
import openai
import numpy as np
from docx import Document
from config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, CHUNK_SIZE

openai.api_key = OPENAI_API_KEY

def load_text(file_path):
    """Loads text from .txt or .docx files."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Splits text into chunks of specified size."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text_list):
    """Generates embeddings using OpenAI API."""
    response = openai.Embedding.create(input=text_list, model=EMBEDDING_MODEL)
    return [e["embedding"] for e in response["data"]]

def store_in_faiss(embeddings):
    """Stores embeddings in a FAISS index."""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)

def process_documents(folder="data/"):
    """Loads, chunks, embeds, and stores all documents in FAISS."""
    texts = []
    for file in os.listdir(folder):
        if file.endswith((".txt", ".docx")):
            text = load_text(os.path.join(folder, file))
            texts.extend(chunk_text(text))
    
    embeddings = embed_text(texts)
    store_in_faiss(embeddings)
    print(f"Processed {len(texts)} chunks and stored embeddings.")

if __name__ == "__main__":
    process_documents()
