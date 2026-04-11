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
MAX_PER_SOURCE = 4 # prevents one source from dominating all results after re-ranking

# Re-ranking
RERANK_CANDIDATE_K = 20  # Retrieve this many candidates before re-ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Temporal decay: score *= e^(-lambda * age_in_patches)
TEMPORAL_LAMBDA = {
    "evergreen": 0.0,
    "mixed": 0.25,
    "version-sensitive": 0.5,
}

# Discrete authority levels (used when discrete_weights mode is enabled)
AUTHORITY_LEVELS = {"low": 0.2, "medium": 0.5, "high": 1.0}

# Evaluation
EVAL_MODEL = "openai/gpt-4o-mini"
