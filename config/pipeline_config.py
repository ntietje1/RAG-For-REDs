from config.settings import DATA_DIR

# Processing (in characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 128

# Embedding
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536          # text-embedding-3-small output dimension

# Vector store — local Qdrant Docker server
# The storage volume is mounted at "data/db" inside the container.
QDRANT_URL = "http://localhost:6333"

# Generation
GENERATION_MODEL = "google/gemini-3-flash-preview"
TOP_K = 5

# Re-ranking
RERANK_CANDIDATE_K = 20  # Retrieve this many candidates before re-ranking

# Temporal decay: score *= e^(-lambda * age_in_patches)
TEMPORAL_LAMBDA = {
    "evergreen": 0.0,
    "mixed": 0.25,
    "version-sensitive": 0.5,
}
