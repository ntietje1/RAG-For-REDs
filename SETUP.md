# Setup & Running the Pipeline

## Prerequisites

**Python 3.10+** and **Docker** are required.

Install the project and its pipeline dependencies:

```bash
pip install -e ".[pipeline]"
```

Copy `.env.example` to `.env` and populate your API key:

```
OPENROUTER_API_KEY=your_key_here
```

---

## Step 1 — Process raw data into chunks

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

## Step 2 — Start the Qdrant server

The pipeline uses a local Qdrant Docker server for significantly better query performance than the embedded Python client. Start it once before running indexing or retrieval:

```bash
docker compose up -d
```

The server mounts `data/db/` as its storage volume, so the index persists across server and machine restarts.

---

## Step 3 — Build the vector index

Embeds the processed chunks and upserts them into the running Qdrant server.

```bash
# Incremental upsert (safe to re-run; existing points are overwritten by stable ID)
python scripts/run_indexing.py

# Full rebuild from scratch
python scripts/run_indexing.py --rebuild
```

| Flag | Default | Description |
|------|---------|-------------|
| `--chunks` | `data/processed/chunks.jsonl` | Input JSONL produced by Step 1 |
| `--index-batch-size` | `500` | Chunks loaded and upserted per iteration — controls peak memory usage |
| `--embed-batch-size` | `100` | Chunks per embedding API request — each index-batch is split into multiple API calls |
| `--rebuild` | off | Clear the index before inserting |

> **Memory note:** with `--index-batch-size 500` and 142k chunks, peak memory is ~3MB per batch instead of ~870MB for the full corpus. Increase `--index-batch-size` to reduce Qdrant upsert round-trips; increase `--embed-batch-size` (up to 2048) only if the API allows larger payloads.

---

## Step 4 — Query the pipeline

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
| `--pipeline` | *(required)* | `baseline` (temporal pipeline not yet implemented) |
| `--query` | — | Single question; omit for interactive REPL |
| `--top-k` | `5` | Number of chunks to retrieve |

---

## Running all steps in sequence

```bash
docker compose up -d && \
python scripts/run_processing.py --raw-dir data && \
python scripts/run_indexing.py --rebuild && \
python scripts/run_retrieval.py --pipeline baseline
```
