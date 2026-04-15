# Improving RAG for Rapidly Evolving Domains Through Temporal Scope Classification and Source Authority Modeling

Standard Retrieval-Augmented Generation (RAG) pipelines assume a relatively stable knowledge corpus, but many real-world domains evolve continuously. Some queries lean heavily on "timeless" information while some queries are inherently very time-sensitive. Current systems naively apply a date filter on retrieved information or ignore temporality entirely. Additionally, both web search and standard RAG rank results by Search Engine Optimization (SEO) signals or embedding similarity scores, neither of which accounts for the fact that in many specialized domains, source type matters significantly. An official patch note, a community wiki entry, or a Reddit thread are not interchangeable even when semantically similar to a query. While frontier Large Language Models (LLMs) with web search handle some of these cases, they do so expensively and without principled mechanisms for temporal reasoning or source evaluation. This project investigates whether lightweight, explicit temporal and authority modeling in the retrieval layer can achieve comparable answer quality at a fraction of the cost, using League of Legends as an evaluation environment. Its frequent gameplay patches create versioned ground truth, its knowledge naturally spans from evergreen to time-sensitive, and it has a diverse ecosystem of sources ranging from official Riot patch notes to meta tracking platforms to community discussions.

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

## Pipeline Architecture

The RAG pipeline is built as a **LangGraph state graph** with the following stages:

```
Query --> [Classify] --> Retrieve --> Rerank --> Generate --> Answer
```

- **Classify** (conditional): An LLM-based classifier determines the query's temporal scope (`evergreen`, `version-sensitive`, `mixed`), per-source authority weights, temporal sensitivity, and generates alternate query phrasings for expansion. Skipped when no classifier-dependent features are enabled.
- **Retrieve**: Embeds the original query and any alternate phrasings, retrieves top-k candidates from Qdrant. Optionally applies a pre-retrieval filter to exclude low-authority sources.
- **Rerank**: Applies a composite scoring formula: `base_score x temporal_decay x authority_weight`, with optional cross-encoder re-ranking. Deduplicates and enforces per-source limits.
- **Generate**: Produces the final answer from the top-k context chunks using an LLM.

The pipeline supports four ablation configurations (baseline, temporal-only, authority-only, full) controlled via constructor flags, with conditional graph edges determining which nodes execute.

### Observability

The pipeline integrates with **LangSmith** for end-to-end tracing. All LangChain operations (LLM calls, embeddings) are auto-traced, and custom pipeline functions (reranking, vector queries, classification) are instrumented with `@traceable` decorators. See [SETUP.md](SETUP.md) for configuration.

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

See [SETUP.md](SETUP.md) for full installation and usage instructions.

> **Quick start — evaluation results only:** To explore pre-computed results without setting up Qdrant or API keys, install dependencies and launch the UI directly:
> ```bash
> pip install -e ".[pipeline]"
> streamlit run app.py
> ```
> Open the **Evaluation Results** tab to view metric breakdowns, charts, and per-question heatmaps from `evaluation/results.json`.

---

## Data & Resources

### Core Knowledge Sources

- Official Data: Riot Games Patch Notes and developer blogs.
- Community Data: The League of Legends Wiki and r/leagueoflegends (subreddit).
- Statistical Data: Data from a statistical aggregator (e.g., U.GG, OP.GG or Lolalytics) to provide objective ground truth for current stats.

### Technical Stack

- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) state graph for pipeline orchestration with conditional node execution.
- **LLM Framework**: [LangChain](https://python.langchain.com/) for LLM and embedding integrations via OpenRouter.
- **Large Language Model**: `gemini-3-flash-preview` via OpenRouter for generation and classification.
- **Embedding Model**: `text-embedding-3-small` via OpenRouter for vectorization.
- **Vector Database**: Qdrant (local Docker server) for storing indexed chunks with metadata.
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers for re-ranking.
- **Evaluation Judge**: `gpt-4o-mini` via OpenRouter (independent model family to avoid self-evaluation bias).
- **Observability**: [LangSmith](https://smith.langchain.com/) for per-node tracing, token usage, and latency monitoring.
- **Routing**: OpenRouter unified API for both embedding and generation calls.

### Evaluation Dataset

- Curated Evaluation Set: A manually constructed dataset of at least 40 questions spanning evergreen, version-sensitive, and source-dependent categories.
