"""
Embedding Model Export
======================
Loads the fine-tuned SentenceTransformer checkpoint and re-exports it in
two formats:

  1. SentenceTransformer native  (.save())      — loadable with
     SentenceTransformer(path) directly; preferred for Python inference.

  2. BertEmbeddingModel wrapper  (state_dict)   — a lightweight nn.Module
     that exposes CLS and mean-pool strategies; useful for serving stacks
     that prefer plain PyTorch.

Both CLS and mean-pool are fully implemented.

✏️ PLACEHOLDER: choose pooling_strategy in ExportConfig (config.py) after
benchmarking both on your retrieval task.  Mean-pool is a good default for
semantic search; CLS is faster and works well after contrastive fine-tuning.

Usage
-----
    python scripts/export_embeddings.py
    make export POOLING=mean
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.Finetuning.config import ExportConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding wrapper (both strategies complete)
# ---------------------------------------------------------------------------
class BertEmbeddingModel(nn.Module):
    """
    Thin wrapper around a BERT encoder that outputs fixed-size sentence
    embeddings using either CLS-token extraction or mean-pooling.

    Parameters
    ----------
    encoder           : BERT backbone loaded from a fine-tuned checkpoint
    pooling_strategy  : "cls" or "mean"
    normalize         : If True, L2-normalise the output vectors
                        (recommended for cosine-similarity search)
    """

    def __init__(
        self,
        encoder: nn.Module,
        pooling_strategy: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if pooling_strategy not in {"cls", "mean"}:
            raise ValueError(
                f"pooling_strategy must be 'cls' or 'mean', got '{pooling_strategy}'"
            )
        self.encoder = encoder
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.hidden_size: int = encoder.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state  # (B, T, H)

        if self.pooling_strategy == "cls":
            # [CLS] token is always at position 0
            embeddings = hidden[:, 0, :]

        else:  # mean pooling
            # Expand attention mask to broadcast over the hidden dimension,
            # then average over real (non-padding) token positions only.
            mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings  # (B, H)

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 256,
        batch_size: int = 64,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Convenience method: tokenise and embed a list of strings in batches.
        Returns a (N, hidden_size) CPU float32 tensor.

        ✏️ PLACEHOLDER: add tqdm progress bar for very large corpora.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()

        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            all_embeddings.append(self(**enc).cpu())

        return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------
def export(cfg: ExportConfig) -> None:
    logger.info("Loading fine-tuned model from %s", cfg.finetuned_model_dir)
    cfg.embedding_model_dir.mkdir(parents=True, exist_ok=True)

    # ── Format 1: SentenceTransformer native save ────────────────────────
    # This is the simplest and most portable format — load with:
    #     model = SentenceTransformer(str(cfg.embedding_model_dir))
    st_model = SentenceTransformer(str(cfg.finetuned_model_dir))
    st_model.save(str(cfg.embedding_model_dir / "sentence_transformer"))
    logger.info(
        "SentenceTransformer model saved → %s",
        cfg.embedding_model_dir / "sentence_transformer",
    )

    # ── Format 2: BertEmbeddingModel (HF backbone + custom pooling) ──────
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.finetuned_model_dir))
    encoder = AutoModel.from_pretrained(str(cfg.finetuned_model_dir))

    embedding_model = BertEmbeddingModel(
        encoder=encoder,
        pooling_strategy=cfg.pooling_strategy,
        normalize=cfg.normalize_embeddings,
    )
    embedding_model.eval()

    # Sanity check: embed two strings and report cosine similarity
    samples = [
        "I'm looking for a romantic Italian restaurant.",
        "Great burgers, fast service, very casual.",
    ]
    embs = embedding_model.encode(samples, tokenizer)
    cos_sim = F.cosine_similarity(embs[0:1], embs[1:2]).item()
    logger.info(
        "Sanity check — embeddings shape=%s  cosine_sim(sample0, sample1)=%.4f",
        tuple(embs.shape),
        cos_sim,
    )

    # Save HF backbone weights and tokeniser (re-loadable with AutoModel)
    encoder.save_pretrained(str(cfg.embedding_model_dir))
    tokenizer.save_pretrained(str(cfg.embedding_model_dir))

    # Save the wrapper's state dict
    state_dict_path = cfg.embedding_model_dir / "embedding_model_state_dict.pt"
    torch.save(embedding_model.state_dict(), state_dict_path)
    logger.info("State dict saved → %s", state_dict_path)

    # Save pooling metadata for downstream loaders
    meta = {
        "pooling_strategy": cfg.pooling_strategy,
        "normalize_embeddings": cfg.normalize_embeddings,
        "hidden_size": embedding_model.hidden_size,
        "finetuned_from": str(cfg.finetuned_model_dir),
    }
    with open(cfg.embedding_model_dir / "embedding_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Export complete → %s", cfg.embedding_model_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export the fine-tuned model as an embedding model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Override ExportConfig.finetuned_model_dir",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override ExportConfig.embedding_model_dir",
    )
    p.add_argument(
        "--pooling_strategy",
        choices=["cls", "mean"],
        default=None,
        help="Override ExportConfig.pooling_strategy",
    )
    p.add_argument("--no_normalize", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = ExportConfig()

    if args.model_dir:       cfg.finetuned_model_dir  = args.model_dir
    if args.output_dir:      cfg.embedding_model_dir  = args.output_dir
    if args.pooling_strategy: cfg.pooling_strategy     = args.pooling_strategy
    if args.no_normalize:    cfg.normalize_embeddings = False

    export(cfg)
