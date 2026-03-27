from config.settings import DATA_DIR

# Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Vector store
VECTOR_STORE_DIR = DATA_DIR / "indices"

# Generation
GENERATION_MODEL = "google/gemini-3-flash-preview"
TOP_K = 5

# Source authority weights (used by temporal-aware retrieval)
SOURCE_AUTHORITY = {
    "riot_patch_notes": 1.0,
    "wiki": 0.8,
    "lolalytics": 0.9,
    "reddit": 0.5,
}
