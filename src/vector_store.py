import faiss
import openai
import numpy as np
from config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K_RESULTS

openai.api_key = OPENAI_API_KEY

def load_faiss_index():
    """Loads the FAISS index from disk."""
    return faiss.read_index(FAISS_INDEX_PATH)

def get_top_matches(query, top_k=TOP_K_RESULTS):
    """Finds the most relevant document chunks."""
    index = load_faiss_index()
    query_embedding = openai.Embedding.create(input=[query], model=EMBEDDING_MODEL)["data"][0]["embedding"]
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    distances, indices = index.search(query_vector, top_k)
    return indices[0]  # Returns top-k matching indices

if __name__ == "__main__":
    query = input("Enter a search query: ")
    print(get_top_matches(query))
