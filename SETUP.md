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

### Restore Qdrant from a snapshot

Use this when you already have a Qdrant snapshot file for the `chunks` collection and want to avoid re-indexing.

1. Start the Qdrant server with `docker compose up -d`.
2. Open the Qdrant dashboard in your browser at `http://localhost:6333/dashboard`.
3. Go to the `Collections` view.
4. If a `chunks` collection already exists and you want to replace it, delete that collection first.
5. Use the snapshot restore / upload action in the dashboard.
6. Select your `chunks.snapshot` file and restore it into a collection named `chunks`.
7. Wait for the restore to complete, then confirm in the dashboard that the `chunks` collection exists and shows a non-zero point count.

If the restored collection already has points, skip indexing and move directly to retrieval.

---

## Step 3 — Build the vector index

Embeds the processed chunks and upserts them into the running Qdrant server.
Skip this step if you restored a valid snapshot and confirmed the collection already has points.

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
# Baseline with query expansion
python scripts/run_retrieval.py --query "What changed for Zeri in patch 25.23?"

# Add enhancements
python scripts/run_retrieval.py --cross-encoder --query "..."
python scripts/run_retrieval.py --temporal --cross-encoder --query "..."
python scripts/run_retrieval.py --temporal --authority --cross-encoder --query "..."

# Use discrete authority levels instead of continuous floats
python scripts/run_retrieval.py --temporal --authority --discrete-weights --query "..."

# Pure baseline — no classifier, no enhancements
python scripts/run_retrieval.py --no-expansion --query "..."
```

**Interactive mode** (omit `--query`):

```bash
python scripts/run_retrieval.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | — | Single question; omit for interactive REPL |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--temporal` | off | Enable temporal decay (down-weight older patches) |
| `--authority` | off | Enable source-authority weighting |
| `--cross-encoder` | off | Enable cross-encoder re-ranking |
| `--no-expansion` | off | Disable query expansion (alternate phrasings) |
| `--discrete-weights` | off | Use discrete authority levels (`low`/`medium`/`high`) instead of continuous floats |

The classifier now outputs additional fields alongside temporal scope and authority weights:
- **reasoning** — chain-of-thought rationale explaining the classification
- **target_patch** — the specific patch version referenced by the query (if any), used as the reference point for temporal decay instead of the latest patch

---

## Step 5 — Run the evaluation

Runs all 30 evaluation questions through 4 ablation configurations (baseline, temporal-only, authority-only, full) and scores each with an LLM-as-judge (GPT-4o-mini) on four metrics: faithfulness, answer relevancy, context precision, and context recall.

```bash
# Full evaluation (all configs, all questions)
python -m evaluation.evaluate

# Quick smoke test (1 question per config)
python -m evaluation.evaluate --dry-run

# Evaluate a subset of configs
python -m evaluation.evaluate --configs baseline full

# Custom paths
python -m evaluation.evaluate --questions evaluation/questions.json --output evaluation/results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--questions` | `evaluation/questions.json` | Path to evaluation questions JSON |
| `--output` | `evaluation/results.json` | Path to write raw per-question results |
| `--configs` | all | Subset of configs to run: `baseline`, `temporal_only`, `authority_only`, `full` |
| `--dry-run` | off | Run only 1 question per config to verify the pipeline works end-to-end |

Results are printed as a comparison table broken down by temporality category (Evergreen, Version-sensitive, Mixed) and saved as JSON for further analysis.

> **Cost note:** a full run makes ~690 API calls (4 configs x 30 questions x ~6 LLM calls each). At Gemini Flash + GPT-4o-mini rates this costs a few cents and takes ~15-20 minutes.

---

## Running all steps in sequence

```bash
docker compose up -d && \
python scripts/run_processing.py --raw-dir data && \
python scripts/run_indexing.py --rebuild && \
python scripts/run_retrieval.py --cross-encoder
```
