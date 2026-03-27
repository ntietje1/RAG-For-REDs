# Improving RAG for Rapidly Evolving Domains Through Temporal Scope Classification and Source Authority Modeling

Standard Retrieval-Augmented Generation (RAG) pipelines assume a relatively stable knowledge corpus, but many real-world domains evolve continuously. Some queries lean heavily on “timeless” information while some queries are inherently very time-sensitive. Current systems naively apply a date filter on retrieved information or ignore temporality entirely. Additionally, both web search and standard RAG rank results by Search Engine Optimization (SEO) signals or embedding similarity scores, neither of which accounts for the fact that in many specialized domains, source type matters significantly. An official patch note, a community wiki entry, or a Reddit thread are not interchangeable even when semantically similar to a query. While frontier Large Language Models (LLMs) with web search handle some of these cases, they do so expensively and without principled mechanisms for temporal reasoning or source evaluation. This project investigates whether lightweight, explicit temporal and authority modeling in the retrieval layer can achieve comparable answer quality at a fraction of the cost, using League of Legends as an evaluation environment. Its frequent gameplay patches create versioned ground truth, its knowledge naturally spans from evergreen to time-sensitive, and it has a diverse ecosystem of sources ranging from official Riot patch notes to meta tracking platforms to community discussions.

## Research Question / Problem Statement

Can a retrieval architecture that models temporal query scope and source authority meaningfully improve answer quality over standard RAG in rapidly-evolving knowledge domains?

## Scope of the Project

In scope:

- Temporal scope classifier for recency weighting
- Source authority mode for retrieval adjustment
- League of Legends as the evaluation domain: Utilization of this domain for its frequent updates, diversity in knowledge sources, and natural spectrum of evergreen vs time-sensitive queries.
- Corpus construction (official Riot Games notes, Wiki, Reddit, Stats)
- Evaluation set (40+ tagged questions along temporal sensitivity and authority category)
- Temporal sensitivity: evergreen, version-sensitive, mixed
- Authority category: official-dependent, community-dependent, cross-source
- Ablation study comparing all components

Out of scope:

- Fine-tuning models based on gathered league of legends knowledge sources and/or retrieval performance.
- Domain generalization experiments on other games or real-world domains, though generalizability will be discussed qualitatively in the final report.
- Deployment as a user-facing product.
- Data pipeline for the constant ingestion of new data sources.
- Comparison to a frontier model with web search.

## Plan of Action

### Week 1: Data Acquisition & Infrastructure

- Perform targeted scraping of official Riot patch notes, the community wiki, and Reddit.
- Set up the Vector Store and define the metadata schema.
- Develop the Embedding Pipeline.
- Develop the 40-question Evaluation Set and define the ground truth for each.

### Week 2: Pipeline Development

- Build the Standard RAG baseline pipeline.
- Implement the Temporal Scope Classifier to detect query time-sensitivity.
- Develop the Source Authority Weighting logic based on source type.

### Week 3: Evaluation & Ablation Study

- Run the evaluation set through four versions: Naive RAG, Temporal-only, Authority-only, and Full Pipeline.
- Compare performance results to quantify the improvement in accuracy.
- Conduct Error Analysis to identify failure points in the weighting logic.
- Document Interesting Failure Cases where the model struggled despite metadata-aware retrieval.

### Week 4: Final Analysis & Documentation

- Create tables and figures visualizing the performance improvements.
- Finalize the technical report and discuss qualitative generalizability.
- Submit the final project documentation and codebase.

## Running the Pipeline

### Prerequisites

Install the pipeline dependencies (requires Python 3.10+):

```bash
pip install -e ".[pipeline]"
```

Copy `.env.example` to `.env` and populate your API keys:

```
OPENROUTER_API_KEY=your_key_here
```

---

### Step 1 — Process raw data into chunks

Reads raw scraped JSON files from `data/`, cleans them, and writes chunked documents to `data/processed/chunks.jsonl`.

```bash
# Process all sources
python scripts/run_processing.py

# Process a single source
python scripts/run_processing.py --source patch_notes
python scripts/run_processing.py --source wiki
python scripts/run_processing.py --source reddit
python scripts/run_processing.py --source stats
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `all` | Which source to process (`patch_notes`, `wiki`, `reddit`, `stats`, `all`) |
| `--raw-dir` | `data/raw` | Root directory of raw scraped JSON files |
| `--output` | `data/processed/chunks.jsonl` | Output JSONL path |

> **Note:** raw data currently lives directly in `data/` (not `data/raw/`). Pass `--raw-dir data` if you have not moved the source directories.

---

### Step 2 — Build the vector index

Embeds the processed chunks and upserts them into a local Qdrant store at `data/indices/`.

```bash
# Incremental upsert (safe to re-run; existing points are overwritten by stable ID)
python scripts/run_indexing.py

# Full rebuild from scratch
python scripts/run_indexing.py --rebuild
```

| Flag | Default | Description |
|------|---------|-------------|
| `--chunks` | `data/processed/chunks.jsonl` | Input JSONL produced by Step 1 |
| `--store-dir` | `data/indices` | Qdrant on-disk store directory |
| `--batch-size` | `100` | Number of chunks per embedding API call |
| `--rebuild` | off | Clear the index before inserting |

---

### Step 3 — Query the pipeline

**Single query:**

```bash
python scripts/run_retrieval.py --pipeline baseline --query "What changed for Zeri in patch 25.23?"
```

**Interactive mode** (omit `--query`):

```bash
python scripts/run_retrieval.py --pipeline baseline
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | *(required)* | `baseline` (temporal not yet implemented) |
| `--query` | — | Single question; omit for interactive REPL |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--store-dir` | `data/indices` | Qdrant store to query against |

---

### Running all steps in sequence

```bash
python scripts/run_processing.py --raw-dir data && \
python scripts/run_indexing.py --rebuild && \
python scripts/run_retrieval.py --pipeline baseline
```

---

## Data & Resources

### Core Knowledge Sources

- Official Data: Riot Games Patch Notes and developer blogs.
- Community Data: The League of Legends Wiki and r/leagueoflegends (subreddit).
- Statistical Data: Data from a statistical aggregator (e.g., U.GG, OP.GG or Lolalytics) to provide objective ground truth for current stats.

### Technical Stack

- Large Language Model: `gemini-2.0-flash` via OpenRouter for generation.
- Embedding Model: `text-embedding-3-small` via OpenRouter for vectorization.
- Vector Database: Qdrant (local on-disk mode) for storing indexed chunks with metadata.
- Routing: OpenRouter unified API for both embedding and generation calls.

### Evaluation Dataset

- Curated Evaluation Set: A manually constructed dataset of at least 40 questions spanning evergreen, version-sensitive, and source-dependent categories.
