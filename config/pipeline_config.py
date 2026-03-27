from config.settings import DATA_DIR

# Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536          # text-embedding-3-small output dimension

# Vector store — local Qdrant Docker server
# The storage volume is mounted at "data/db" inside the container.
QDRANT_URL = "http://localhost:6333"

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
