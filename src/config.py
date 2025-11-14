import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./embeddings/chroma_db")


HF_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
GROQ_API_URL = "https://api.groq.com/openai/v1"   