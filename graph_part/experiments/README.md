# Chat Conversation Graph Builder

Build a similarity graph from ChatGPT-style chat histories with multilingual preprocessing, keyword extraction, clustering, and graph metadata.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the CLI against a chat history JSON file (array of `{id, role, content, timestamp}` objects). The loader also understands ChatGPT export payloads that expose a `mapping` structure (see `../input_data/mock_data.json`).

```bash
python build_graph.py --in sample_history.json --out graph.json --cfg config.yaml
```

To process a different conversation, point `--in` at your own history (for example, `../input_data/mock_data.json`) and reuse or customize the config.

The script prints a concise summary and writes a validated `graph.json` alongside full metadata.

## Configuration

`config.yaml` captures all tunable parameters:

- `embedding_model`: Sentence-Transformers model name. Defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `keyword`: KeyBERT extraction knobs (`top_n`, `max_ngram`, `dedup_thresh`).
- `graph`: Similarity graph strategy (`sim_top_k` or `sim_threshold`).
- `cluster`: HDBSCAN parameters (`min_cluster_size`, `min_samples`, `metric`).
- `preprocess`: Text cleaning controls, URL/code stripping, casing, punctuation, and stopword languages.

Set values in a copy of the YAML file and pass it to `--cfg`.

## Architecture

1. **Input validation** – `io_schemas.py` provides Pydantic models to enforce the chat schema and final output contract.
2. **Preprocessing** – configurable cleaning removes URLs and fenced code, lowercases text, builds multilingual stoplists, and chunks long messages (~512 chars).
3. **Embeddings & keywords** – Sentence-Transformers (with deterministic fallback) create normalized embeddings; KeyBERT extracts deduplicated keywords per message.
4. **Clustering & summaries** – HDBSCAN identifies topical clusters; a TF-IDF pass over cluster text surfaces top descriptive terms.
5. **Graph construction** – cosine similarities define edges (top-k or threshold), producing `graph.json` with nodes, edges, and metadata (counts, params, clusters).

## Tests

After installation, run:

```bash
pytest
```

Tests execute the pipeline on `sample_history.json`, assert schema compliance, and validate graph density under alternate settings. They remain lightweight (<10s).
