import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
METADATA_FILE = DATA_DIR / "metadata.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
CHROMA_DIR = str(BASE_DIR / "db")
COLLECTION_NAME = "papers"

# Data collection
ARXIV_QUERY = 'retrieval augmented generation'
MAX_PAPERS = 50

# Chunking (characters, roughly 400 tokens = 1600 chars)
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 100

# Models
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = "claude-sonnet-5-5"

# Retrieval / trust
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 6
ABSTAIN_THRESHOLD = 0.4

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")