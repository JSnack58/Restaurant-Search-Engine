"""
Training configuration
======================
All tuneable hyper-parameters and infrastructure settings live here as
dataclasses.  train_contrastive.py, evaluate.py, and export_embeddings.py
import these directly; override individual fields via CLI args in each script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ESConfig:
    """ElasticSearch connection and index names."""

    # ✏️ PLACEHOLDER: override with ES_HOST env var for a remote or secured cluster.
    # For Docker Compose local dev, the default http://localhost:9200 works as-is.
    host: str = "http://localhost:9200"

    # Raw business data (written by es_ingest_businesses.py)
    businesses_raw_index: str = "yelp_businesses_raw"

    # Processed training pairs (written by es_process.py)
    training_pairs_index: str = "yelp_training_pairs"

    # Rows fetched per ES search request (not the same as the DataLoader batch size)
    es_batch_size: int = 256

    # In-memory shuffle buffer size inside ElasticSearchDataset
    shuffle_buffer: int = 5_000

    # Cap the number of samples streamed per epoch (None = all)
    max_samples: int | None = 100_000
    max_val_samples: int | None = 10_000

    # How long to keep the Point-in-Time context alive between requests
    pit_keep_alive: str = "2m"


@dataclass
class ModelConfig:
    # ✏️ PLACEHOLDER: swap to a domain-adapted or lighter checkpoint, e.g.:
    #   "sentence-transformers/all-MiniLM-L6-v2"   (fast, 384-dim)
    #   "sentence-transformers/all-mpnet-base-v2"   (higher quality, 768-dim)
    pretrained_model_name: str = "bert-base-uncased"


@dataclass
class TrainingConfig:
    # Where checkpoints and logs are written
    output_dir: Path = Path("models/checkpoints")

    # ── Epochs & batch ────────────────────────────────────────────────────
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64

    # ── Optimiser ─────────────────────────────────────────────────────────
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06      # fraction of total steps used for linear warm-up

    # Set True to enable ReduceLROnPlateau (drops LR when val_loss stalls)
    use_lr_schedule: bool = True
    lr_reduce_factor: float = 0.5   # multiply LR by this when plateau is hit
    lr_reduce_patience: int = 1     # epochs with no improvement before reducing

    # ── Early stopping ─────────────────────────────────────────────────────
    early_stopping_patience: int = 2

    # ── Hardware ──────────────────────────────────────────────────────────
    # ✏️ PLACEHOLDER: set fp16=True for GPUs with Tensor Cores (Volta+).
    fp16: bool = False

    # Keep at 0 for a local single-node ES instance.
    # ✏️ PLACEHOLDER: if tokenization becomes the CPU bottleneck (profile first!),
    #   increase num_workers AND add get_worker_info() sharding to ElasticSearchDataset.
    dataloader_num_workers: int = 0

    # ── Logging ───────────────────────────────────────────────────────────
    logging_steps: int = 100

    # ✏️ PLACEHOLDER: add "wandb" and set the WANDB_PROJECT environment variable
    #   before running make train.  e.g.: export WANDB_PROJECT=restaurant-search
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])

    run_name: str = "bert-restaurant-contrastive"
    seed: int = 42


@dataclass
class ExportConfig:
    finetuned_model_dir: Path = Path("models/checkpoints/best_model")
    embedding_model_dir: Path = Path("models/embedding_model")

    # "cls"  — use the [CLS] token (fast, good baseline)
    # "mean" — mean-pool all token hidden states (often higher retrieval quality)
    # ✏️ PLACEHOLDER: benchmark both on a held-out retrieval task before deciding.
    pooling_strategy: str = "mean"

    # L2-normalise output vectors (recommended for cosine-similarity search)
    normalize_embeddings: bool = True
