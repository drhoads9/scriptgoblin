import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key. Set it in the .env file or environment variables.")

EMBEDDING_MODEL = "text-embedding-ada-002"
FAISS_INDEX_PATH = "data/vector_index"
CHUNK_SIZE = 500
TOP_K_RESULTS = 3
