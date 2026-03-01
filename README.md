# Restaurant Search Engine — BERT Contrastive Training Pipeline

A reproducible pipeline for fine-tuning `bert-base-uncased` on the
[Yelp Open Dataset](https://www.yelp.com/dataset) using contrastive learning
(MNRL and Triplet Loss), then serving the result as a semantic restaurant
search engine backed by Elasticsearch kNN.

---

## How it works

```
yelp_academic_dataset_business.json
            │
            │  make ingest_raw
            ▼
  ElasticSearch: yelp_businesses_raw
  (business_id, name, categories, attributes, city, state)
            │
            │  make process_data
            │  BusinessDTO  ← parses & normalises raw business records
            │  build_business_text()
            │  generate_queries()
            │  assign_split()
            ▼
  ElasticSearch: yelp_training_pairs
  (query, business_text, split, shuffle_id)
            │
            │  make train   [LOSS_TYPE=mnrl|triplet]
            ▼
  SentenceTransformer checkpoint
  models/checkpoints/best_model/
            │
            │  make evaluate
            │  EmbeddingSimilarity (Spearman) + TripletAccuracy
            │
            │  make export
            ▼
  models/embedding_model/sentence_transformer/
            │
            │  make index_embeddings
            ▼
  ElasticSearch: yelp_businesses_knn
  (dense_vector HNSW index, dims auto-detected from model)
            │
            │  make search
            ▼
  Interactive terminal REPL
  (encode query → kNN search → ranked results with optional city/state filter)
```

Elasticsearch is the **sole storage layer** — no CSV or Parquet files.
Data flows from raw business records through synthetic query generation,
contrastive fine-tuning, and into a kNN search index that the interactive
search REPL queries in real time.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| Docker + Docker Compose | any recent version |
| Yelp Open Dataset | `yelp_academic_dataset_business.json` |

---

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd Restaurant-Search-Engine
make setup
```

### 2. Download the Yelp dataset

Sign up and download from <https://www.yelp.com/dataset>.
Place the business file at:

```
data/raw/yelp_academic_dataset_business.json
```

### 3. Start ElasticSearch

```bash
make es_start
```

This runs a single-node ES 8.x container on `http://localhost:9200` with
security disabled (suitable for local development).

Verify it is healthy:

```bash
curl http://localhost:9200/_cluster/health
```

### 4. Ingest and process

```bash
make ingest_raw       # streams JSON → yelp_businesses_raw
make process_data     # generates (query, business_text) pairs → yelp_training_pairs
```

`process_data` uses `BusinessDTO` to parse each business, `build_business_text()`
to compose a natural-language description, `generate_queries()` to produce
synthetic search queries, and `assign_split()` to deterministically partition
into train / val / test by `business_id`.

### 5. Train

```bash
# Multiple Negatives Ranking Loss (default)
make train

# Triplet Loss
make train LOSS_TYPE=triplet

# Custom hyperparameters
make train LOSS_TYPE=mnrl EPOCHS=5 BATCH_SIZE=64

# Continue from an existing checkpoint
python scripts/Finetuning/train_contrastive.py \
    --loss_type mnrl \
    --resume_from models/checkpoints/best_model
```

Checkpoints are saved to `models/checkpoints/checkpoint-epoch-N/`.
The best model (lowest validation loss) is written to
`models/checkpoints/best_model/`.

Training stops early when:
- Validation loss does not improve for `early_stopping_patience` epochs, **or**
- Either train or val loss drops below `early_stopping_threshold`

### 6. Evaluate

```bash
make evaluate                  # val split (default)
make evaluate EVAL_SPLIT=test  # held-out test split
```

Reports two metrics:

- **EmbeddingSimilarity** — Spearman correlation between predicted cosine
  similarity and ground-truth positive/negative labels.
- **TripletAccuracy** — fraction of triplets where the anchor query is
  closer to its matching business than to the sampled negative.

Results are saved to `models/checkpoints/best_model/eval_<split>_results.json`.

### 7. Export the embedding model

```bash
make export              # mean-pool (default)
make export POOLING=cls  # CLS-token pooling
```

Writes two artefacts to `models/embedding_model/`:

| File | How to load |
|---|---|
| `sentence_transformer/` | `SentenceTransformer("models/embedding_model/sentence_transformer")` |
| `embedding_model_state_dict.pt` | `torch.load(...)` into a `BertEmbeddingModel` instance |

### 8. Build the kNN search index

```bash
make index_embeddings
```

Reads all restaurants from `yelp_businesses_raw`, encodes each business text
with the exported `SentenceTransformer`, and bulk-indexes the resulting vectors
into `yelp_businesses_knn` using ES `dense_vector` + HNSW for approximate
nearest-neighbour search.

To rebuild the index from scratch:

```bash
python scripts/DataHandling/es_index_embeddings.py --recreate
```

### 9. Search

```bash
make search
```

Starts an interactive REPL. Enter a natural-language query and optional
city / state filters to retrieve the top-10 most semantically similar
restaurants:

```
Search: romantic Italian dinner
City filter   (blank=all): Las Vegas
State filter  (blank=all):

   1. Bella Italia — Las Vegas, NV  [score: 0.9123]
      "Bella Italia — Italian, Pizza — romantic atmosphere — Las Vegas, NV"
   2. ...

Search: cheap tacos open late
City filter   (blank=all):
...

Search: quit
```

Type `quit` or press `Ctrl-C` to exit.

---

## Makefile reference

| Target | Description | Key variables |
|---|---|---|
| `make setup` | Install Python dependencies | — |
| `make es_start` | Start self-hosted ES via Docker Compose | `ES_HOST` |
| `make es_stop` | Stop ES container | — |
| `make ingest_raw` | Business JSON → ES raw index | `BIZ_INDEX` |
| `make process_data` | ES raw → training pairs index | `PAIRS_INDEX` |
| `make train` | Contrastive fine-tuning | `EPOCHS` `BATCH_SIZE` `LOSS_TYPE` |
| `make evaluate` | Embedding quality metrics | `EVAL_SPLIT` |
| `make export` | Save embedding model | `POOLING` |
| `make index_embeddings` | Embed all businesses → ES kNN index | `KNN_INDEX` |
| `make search` | Interactive kNN search REPL | `KNN_INDEX` |
| `make lint` | Lint `scripts/` (ruff + black --check) | — |
| `make format` | Auto-format `scripts/` in place | — |

---

## Configuration

All hyperparameters live in
[scripts/Finetuning/config.py](scripts/Finetuning/config.py) as dataclasses.

```python
# ── ElasticSearch ──────────────────────────────────────────────────────────
ESConfig.host                    # default: http://localhost:9200
ESConfig.max_samples             # train docs per epoch cap  (default: 100_000)
ESConfig.max_val_samples         # val docs per epoch cap    (default: 10_000)

# ── Model ──────────────────────────────────────────────────────────────────
ModelConfig.pretrained_model_name  # default: bert-base-uncased

# ── Training ───────────────────────────────────────────────────────────────
TrainingConfig.num_train_epochs        # default: 3
TrainingConfig.learning_rate           # default: 2e-5
TrainingConfig.fp16                    # set True for Tensor Core GPUs
TrainingConfig.use_lr_schedule         # ReduceLROnPlateau  (default: True)
TrainingConfig.lr_reduce_factor        # LR multiplier on plateau  (default: 0.5)
TrainingConfig.lr_reduce_patience      # epochs before reducing LR (default: 1)
TrainingConfig.early_stopping_patience # epochs before stopping    (default: 2)
TrainingConfig.early_stopping_threshold # stop if loss < this      (default: 0.1)
TrainingConfig.report_to               # ["tensorboard"] or ["wandb"]

# ── Export ─────────────────────────────────────────────────────────────────
ExportConfig.pooling_strategy    # "cls" or "mean"
```

### TensorBoard

```bash
python -m tensorboard.main --logdir models/checkpoints/runs
```

### W&B logging

```bash
pip install wandb
export WANDB_PROJECT=restaurant-search
make train REPORT_TO=wandb
```

---

## Project structure

```
Restaurant-Search-Engine/
├── data/
│   └── raw/                             ← place yelp_academic_dataset_business.json here
├── models/
│   ├── checkpoints/                     ← training checkpoints + best_model/ (auto-created)
│   └── embedding_model/                 ← exported model (auto-created)
├── plots/                               ← embedding scatter plots (auto-created)
├── scripts/
│   ├── DataHandling/
│   │   ├── business_dto.py              ← BusinessDTO: typed, normalised business record
│   │   ├── constants.py                 ← attribute + category lists
│   │   ├── templates.py                 ← query template strings
│   │   ├── es_ingest_businesses.py      ← ingest raw business data into ES
│   │   ├── es_process.py               ← generate training pairs from ES
│   │   └── es_index_embeddings.py      ← encode businesses → ES kNN index
│   ├── Finetuning/
│   │   ├── config.py                   ← all hyperparameters
│   │   ├── es_dataset.py               ← ES-backed IterableDataset
│   │   ├── es_pairs_dataset.py         ← InputExample pair / triplet builder
│   │   └── train_contrastive.py        ← training loop (MNRL + Triplet Loss)
│   ├── Evaluation/
│   │   └── evaluate.py                 ← EmbeddingSimilarity + TripletAccuracy
│   ├── export_embeddings.py            ← CLS / mean-pool model export
│   ├── search.py                       ← interactive kNN search REPL
│   ├── sanity_check.py                 ← quick top-3 retrieval check
│   └── plot_embeddings.py              ← t-SNE / PCA embedding scatter plot
├── docker-compose.yml                   ← self-hosted ElasticSearch
├── Makefile
└── requirements.txt
```

---

## ElasticSearch indices

| Index | Written by | Contents |
|---|---|---|
| `yelp_businesses_raw` | `make ingest_raw` | Original business records |
| `yelp_training_pairs` | `make process_data` | `(query, business_text, split, shuffle_id)` |
| `yelp_businesses_knn` | `make index_embeddings` | `(business_text, embedding, name, city, state, categories)` |

The training DataLoader streams directly from `yelp_training_pairs` using the ES
[Point-in-Time + search_after](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html)
pattern — memory usage is O(shuffle_buffer), not O(dataset size).

The kNN index uses ES `dense_vector` with HNSW (`"index": true`) for
sub-linear approximate nearest-neighbour search at query time.

---

## Stopping ElasticSearch

```bash
make es_stop
```

Business and training data persisted in the `esdata` Docker volume survive
restarts. To wipe the volume and start fresh:

```bash
docker compose down -v
```
