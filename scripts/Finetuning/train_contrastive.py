"""
Contrastive Training Loop — Scaffold
======================================
Wires up model loading, DataLoader, AdamW optimiser, linear LR scheduler,
epoch loop, checkpointing, early stopping, and TensorBoard/W&B logging.

The team must implement the two loss functions below:
    compute_mnrl_loss()    — Multiple Negatives Ranking Loss
    compute_triplet_loss() — Triplet Loss with margin

Usage
-----
    python scripts/Finetuning/train_contrastive.py                  # MNRL default
    python scripts/Finetuning/train_contrastive.py --loss_type triplet --num_epochs 5
    make train LOSS_TYPE=triplet EPOCHS=5

Checkpoints are saved to models/checkpoints/checkpoint-epoch-N/.
The best model (lowest val loss) is saved to models/checkpoints/best_model/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.Finetuning.config import ESConfig, ModelConfig, TrainingConfig
from scripts.Finetuning.es_pairs_dataset import ESPairsDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ✏️  TEAM STUBS — implement these two functions
# ---------------------------------------------------------------------------

def compute_mnrl_loss(
    queries_emb: torch.Tensor,
    businesses_emb: torch.Tensor,
) -> torch.Tensor:
    """
    ✏️ PLACEHOLDER — Implement Multiple Negatives Ranking Loss.

    Parameters
    ----------
    queries_emb    : (B, H) query embeddings with gradients attached
    businesses_emb : (B, H) business-text embeddings with gradients attached

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Implement compute_mnrl_loss() in scripts/Finetuning/train_contrastive.py"
    )


def compute_triplet_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_emb: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    ✏️ PLACEHOLDER — Implement Triplet Loss with margin.

    Parameters
    ----------
    anchor_emb   : (B, H) anchor (query) embeddings
    positive_emb : (B, H) positive (matching business) embeddings
    negative_emb : (B, H) negative (non-matching business) embeddings
    margin       : Minimum desired gap between positive and negative distances

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Implement compute_triplet_loss() in scripts/Finetuning/train_contrastive.py"
    )


# ---------------------------------------------------------------------------
# Glue: embedding helper
# ---------------------------------------------------------------------------
def _encode(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Tokenise and forward-pass `texts`, preserving the gradient graph.
    Returns a (len(texts), hidden_dim) tensor on `device`.
    """
    features = model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items() if hasattr(v, "to")}
    output = model(features)
    return output["sentence_embedding"]  # (B, H)


# ---------------------------------------------------------------------------
# Glue: training entry point
# ---------------------------------------------------------------------------
def train(
    es_cfg: ESConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    loss_type: str,
) -> None:
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | loss_type: %s", device, loss_type)

    # ── TensorBoard ──────────────────────────────────────────────────────
    tb_writer: SummaryWriter | None = None
    if "tensorboard" in train_cfg.report_to:
        log_dir = train_cfg.output_dir / "runs" / train_cfg.run_name
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard logs → %s", log_dir)

    # ✏️ PLACEHOLDER: initialise W&B if "wandb" in train_cfg.report_to:
    #   import wandb
    #   wandb.init(project=os.getenv("WANDB_PROJECT"), name=train_cfg.run_name)

    # ── Model ────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", model_cfg.pretrained_model_name)
    model = SentenceTransformer(model_cfg.pretrained_model_name)
    model.to(device)

    # ── Datasets & DataLoaders ───────────────────────────────────────────
    train_dataset = ESPairsDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split="train",
        mode=loss_type,
        es_batch_size=es_cfg.es_batch_size,
        shuffle_buffer=es_cfg.shuffle_buffer,
    )
    val_dataset = ESPairsDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split="val",
        mode=loss_type,
        es_batch_size=es_cfg.es_batch_size,
        shuffle_buffer=1_000,
    )

    def _collate(batch):
        return batch  # keep as list[InputExample]; _encode() handles tokenization

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.per_device_train_batch_size,
        num_workers=train_cfg.dataloader_num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.per_device_eval_batch_size,
        num_workers=train_cfg.dataloader_num_workers,
        collate_fn=_collate,
    )

    # ── Optimiser & LR scheduler ─────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ✏️ PLACEHOLDER: replace the estimated steps_per_epoch with the exact
    # value once you know the training set size: total_pairs / batch_size.
    _STEPS_PER_EPOCH_ESTIMATE = 1_000
    total_steps = _STEPS_PER_EPOCH_ESTIMATE * train_cfg.num_train_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if (train_cfg.fp16 and torch.cuda.is_available())
        else None
    )

    # ── Loss dispatcher ───────────────────────────────────────────────────
    def _step_loss(batch) -> torch.Tensor:
        if loss_type == "mnrl":
            q_emb = _encode(model, [ex.texts[0] for ex in batch], device)
            b_emb = _encode(model, [ex.texts[1] for ex in batch], device)
            return compute_mnrl_loss(q_emb, b_emb)
        else:  # triplet
            a_emb = _encode(model, [ex.texts[0] for ex in batch], device)
            p_emb = _encode(model, [ex.texts[1] for ex in batch], device)
            n_emb = _encode(model, [ex.texts[2] for ex in batch], device)
            return compute_triplet_loss(a_emb, p_emb, n_emb)

    # ── Training loop ────────────────────────────────────────────────────
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, train_cfg.num_train_epochs + 1):

        # ── Train epoch ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = _step_loss(batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = _step_loss(batch)
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if step % train_cfg.logging_steps == 0:
                avg = epoch_loss / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d | step %d | train_loss=%.4f | lr=%.2e",
                    epoch, step, avg, lr_now,
                )
                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg, global_step)
                    tb_writer.add_scalar("train/lr", lr_now, global_step)
                # ✏️ PLACEHOLDER: wandb.log({"train/loss": avg, "train/lr": lr_now})

        # ── Validation epoch ─────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                val_loss_sum += _step_loss(batch).item()
                val_steps += 1

        avg_val = val_loss_sum / max(val_steps, 1)
        logger.info("Epoch %d | val_loss=%.4f", epoch, avg_val)
        if tb_writer:
            tb_writer.add_scalar("val/loss", avg_val, epoch)
        # ✏️ PLACEHOLDER: wandb.log({"val/loss": avg_val})

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt_dir = train_cfg.output_dir / f"checkpoint-epoch-{epoch}"
        model.save(str(ckpt_dir))
        logger.info("Checkpoint → %s", ckpt_dir)

        # ── Early stopping ───────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_dir = train_cfg.output_dir / "best_model"
            model.save(str(best_dir))
            logger.info("New best (val_loss=%.4f) → %s", best_val_loss, best_dir)
        else:
            patience_counter += 1
            logger.info(
                "No improvement. Patience: %d/%d",
                patience_counter,
                train_cfg.early_stopping_patience,
            )
            if patience_counter >= train_cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    if tb_writer:
        tb_writer.close()

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive fine-tuning scaffold for restaurant embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--loss_type",
        choices=["mnrl", "triplet"],
        default="mnrl",
    )
    p.add_argument("--model_dir",  type=Path,  help="Override checkpoint output dir")
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr",         type=float)
    p.add_argument("--fp16",       action="store_true")
    p.add_argument(
        "--report_to",
        nargs="+",
        choices=["tensorboard", "wandb", "none"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    es_cfg    = ESConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if args.model_dir:  train_cfg.output_dir = args.model_dir
    if args.num_epochs: train_cfg.num_train_epochs = args.num_epochs
    if args.batch_size: train_cfg.per_device_train_batch_size = args.batch_size
    if args.lr:         train_cfg.learning_rate = args.lr
    if args.fp16:       train_cfg.fp16 = True
    if args.report_to:  train_cfg.report_to = args.report_to

    train(es_cfg, model_cfg, train_cfg, loss_type=args.loss_type)
