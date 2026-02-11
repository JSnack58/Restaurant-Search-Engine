# Restaurant Search Engine — BERT Contrastive Training Pipeline

A reproducible pipeline for fine-tuning `bert-base-uncased` on the
[Yelp Open Dataset](https://www.yelp.com/dataset) using contrastive learning
(MNRL and Triplet Loss), then exporting the result as a high-performance
embedding model for semantic restaurant search.

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
            │  build_business_text()  ← team implements
            │  generate_queries()     ← team implements
            │  assign_split()         ← team implements
            ▼
  ElasticSearch: yelp_training_pairs
  (query, business_text, split, shuffle_id)
            │
            │  make train
            ▼
  SentenceTransformer (bert-base-uncased)
  trained with MNRL or Triplet Loss    ← team implements the losses
            │
            │  make export
            ▼
  models/embedding_model/
  (SentenceTransformer native + BertEmbeddingModel state dict)
```

ElasticSearch is the **sole storage layer** — no CSV or Parquet files.
Data flows from raw business records through synthetic query generation
and into a training-pairs index that the PyTorch DataLoader streams
directly during training.

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

### 4. Implement the three team stubs

Before the data pipeline can run, open
[scripts/DataHandling/es_process.py](scripts/DataHandling/es_process.py)
and fill in the three `NotImplementedError` functions:

| Function | What to write |
|---|---|
| `stream_and_index()` in [es_ingest_businesses.py](scripts/DataHandling/es_ingest_businesses.py) | Read the JSON file line-by-line and bulk-index into ES |
| `build_business_text(business)` | Compose a text description from name, categories, and attributes |
| `generate_queries(business)` | Return a list of synthetic user queries for that business |
| `assign_split(business_id)` | Return `"train"`, `"val"`, or `"test"` deterministically by `business_id` |

### 5. Ingest and process

```bash
make ingest_raw       # streams JSON → yelp_businesses_raw
make process_data     # generates (query, business_text) pairs → yelp_training_pairs
```

### 6. Implement the loss functions

Open [scripts/Finetuning/train_contrastive.py](scripts/Finetuning/train_contrastive.py)
and implement:

| Function | Signature |
|---|---|
| `compute_mnrl_loss` | `(queries_emb: Tensor, businesses_emb: Tensor) → Tensor` |
| `compute_triplet_loss` | `(anchor: Tensor, positive: Tensor, negative: Tensor, margin) → Tensor` |

Both receive `(B, H)` embedding tensors with gradients attached.
The training loop, backprop, optimiser step, logging, and checkpointing are
already wired up.

### 7. Train

```bash
# Multiple Negatives Ranking Loss (default)
make train

# Triplet Loss
make train LOSS_TYPE=triplet

# Custom hyperparameters
make train LOSS_TYPE=mnrl EPOCHS=5 BATCH_SIZE=64 REPORT_TO=wandb
```

Checkpoints are saved to `models/checkpoints/checkpoint-epoch-N/`.
The best model (lowest validation loss) is written to
`models/checkpoints/best_model/` and triggers early stopping after
`early_stopping_patience` epochs without improvement.

### 8. Evaluate

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

### 9. Export the embedding model

```bash
make export              # mean-pool (default)
make export POOLING=cls  # CLS-token pooling
```

Writes two artefacts to `models/embedding_model/`:

| File | How to load |
|---|---|
| `sentence_transformer/` | `SentenceTransformer("models/embedding_model/sentence_transformer")` |
| `embedding_model_state_dict.pt` | `torch.load(...)` into a `BertEmbeddingModel` instance |

---

## Makefile reference

| Target | Description | Key variables |
|---|---|---|
| `make setup` | Install Python dependencies | — |
| `make es_start` | Start self-hosted ES via Docker Compose | `ES_HOST` |
| `make es_stop` | Stop ES container | — |
| `make ingest_raw` | Business JSON → ES raw index | `BIZ_INDEX` |
| `make process_data` | ES raw → training pairs index | `PAIRS_INDEX` |
| `make train` | Contrastive fine-tuning | `EPOCHS` `BATCH_SIZE` `LOSS_TYPE` `REPORT_TO` |
| `make evaluate` | Embedding quality metrics | `EVAL_SPLIT` |
| `make export` | Save embedding model | `POOLING` |
| `make lint` | Lint `scripts/` (ruff + black --check) | — |
| `make format` | Auto-format `scripts/` in place | — |

---

## Configuration

All hyperparameters live in
[scripts/Finetuning/config.py](scripts/Finetuning/config.py) as dataclasses.
Override at import time in each script, or via CLI flags.

```python
# Key settings to review before training
ESConfig.host                    # default: http://localhost:9200
ModelConfig.pretrained_model_name # default: bert-base-uncased
TrainingConfig.num_train_epochs  # default: 3
TrainingConfig.learning_rate     # default: 2e-5
TrainingConfig.fp16              # set True for Tensor Core GPUs
TrainingConfig.report_to         # ["tensorboard"] or ["wandb"]
ExportConfig.pooling_strategy    # "cls" or "mean"
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
│   └── raw/                         ← place yelp_academic_dataset_business.json here
├── models/
│   ├── checkpoints/                 ← training checkpoints (auto-created)
│   └── embedding_model/             ← exported model (auto-created)
├── scripts/
│   ├── DataHandling/
│   │   ├── es_ingest_businesses.py  ← ingest raw business data into ES
│   │   └── es_process.py           ← generate training pairs from ES
│   ├── Finetuning/
│   │   ├── config.py               ← all hyperparameters
│   │   ├── es_dataset.py           ← ES-backed IterableDataset
│   │   ├── es_pairs_dataset.py     ← InputExample pair/triplet builder
│   │   └── train_contrastive.py    ← training loop scaffold
│   ├── Evaluation/
│   │   └── evaluate.py             ← embedding quality metrics
│   └── export_embeddings.py        ← CLS / mean-pool model export
├── docker-compose.yml               ← self-hosted ElasticSearch
├── Makefile
└── requirements.txt
```

---

## ElasticSearch indices

| Index | Written by | Contents |
|---|---|---|
| `yelp_businesses_raw` | `make ingest_raw` | Original business records |
| `yelp_training_pairs` | `make process_data` | `(query, business_text, split, shuffle_id)` |

The DataLoader streams directly from `yelp_training_pairs` using the ES
[Point-in-Time + search_after](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html)
pattern — memory usage is O(shuffle_buffer), not O(dataset size).

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