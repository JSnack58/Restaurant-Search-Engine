"""
Embedding Quality Evaluation
==============================
Loads the best fine-tuned model and evaluates embedding quality on the
val or test split using two sentence-transformers evaluators:

  EmbeddingSimilarityEvaluator — Spearman/Pearson correlation between
      predicted cosine similarity and ground-truth binary labels
      (1.0 for a matched query-business pair, 0.0 for a sampled negative).

  TripletEvaluator — fraction of (anchor, positive, negative) triplets
      where the anchor is closer to the positive than the negative.

✏️ PLACEHOLDER: add downstream retrieval metrics (NDCG@k, MRR, Recall@k)
once you have a retrieval benchmark or a held-out query set.

Usage
-----
    python scripts/Evaluation/evaluate.py                    # val split
    python scripts/Evaluation/evaluate.py --split test
    make evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.Finetuning.config import ESConfig
from scripts.Finetuning.es_dataset import ElasticSearchDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: build evaluator inputs from ES data
# ---------------------------------------------------------------------------
def _load_pairs(
    es_cfg: ESConfig,
    split: str,
    max_samples: int,
) -> tuple[list[str], list[str], list[float]]:
    """
    Stream `max_samples` docs from the clean index, label positive pairs
    as 1.0, then sample an equal number of random negatives labelled 0.0.

    Returns (sentences1, sentences2, scores) for EmbeddingSimilarityEvaluator.
    """
    dataset = ElasticSearchDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split=split,
        es_batch_size=256,
        shuffle_buffer=max_samples,
    )

    docs = []
    for doc in dataset:
        docs.append(doc)
        if len(docs) >= max_samples * 2:  # over-fetch for negative sampling
            break

    random.shuffle(docs)
    positives = docs[:max_samples]

    sentences1, sentences2, scores = [], [], []

    # Positive pairs
    for doc in positives:
        sentences1.append(doc["query"])
        sentences2.append(doc["business_text"])
        scores.append(1.0)

    # Negative pairs — pair each query with a different business
    negatives = docs[max_samples : max_samples * 2]
    for pos, neg in zip(positives, negatives):
        sentences1.append(pos["query"])
        sentences2.append(neg["business_text"])
        scores.append(0.0)

    return sentences1, sentences2, scores


def _load_triplets(
    es_cfg: ESConfig,
    split: str,
    max_samples: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Build (anchor_query, positive_biz, negative_biz) triplets from ES.
    Returns (anchors, positives, negatives) for TripletEvaluator.
    """
    dataset = ElasticSearchDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split=split,
        es_batch_size=256,
        shuffle_buffer=max_samples * 2,
    )

    docs = []
    for doc in dataset:
        docs.append(doc)
        if len(docs) >= max_samples * 2:
            break

    random.shuffle(docs)
    n = min(max_samples, len(docs) // 2)
    positives_pool = docs[:n]
    negatives_pool = docs[n : n * 2]

    anchors, positives, negatives = [], [], []
    for pos, neg in zip(positives_pool, negatives_pool):
        anchors.append(pos["query"])
        positives.append(pos["business_text"])
        negatives.append(neg["business_text"])

    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(
    model_dir: Path,
    es_cfg: ESConfig,
    split: str,
    max_samples: int,
    output_dir: Path,
) -> dict:
    logger.info("Loading model from %s", model_dir)
    model = SentenceTransformer(str(model_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ── Similarity evaluator ─────────────────────────────────────────────
    logger.info("Building similarity pairs (split=%s, max=%d)…", split, max_samples)
    s1, s2, scores = _load_pairs(es_cfg, split, max_samples)

    sim_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=s1,
        sentences2=s2,
        scores=scores,
        name=f"{split}_similarity",
        show_progress_bar=True,
    )
    sim_score = sim_evaluator(model, output_path=str(output_dir))
    spearman = sim_score.get(f"{split}_similarity_spearman_cosine", next(iter(sim_score.values())))
    results["similarity_spearman"] = spearman
    logger.info("EmbeddingSimilarity (Spearman): %.4f", spearman)

    # ── Triplet evaluator ─────────────────────────────────────────────────
    logger.info("Building triplets…")
    anchors, positives, negatives = _load_triplets(es_cfg, split, max_samples)

    triplet_evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name=f"{split}_triplets",
        show_progress_bar=True,
    )
    triplet_acc = triplet_evaluator(model, output_path=str(output_dir))
    accuracy = triplet_acc.get(f"{split}_triplets_cosine_accuracy", next(iter(triplet_acc.values())))
    results["triplet_accuracy"] = accuracy
    logger.info("TripletEvaluator accuracy: %.4f", accuracy)

    # ✏️ PLACEHOLDER — downstream retrieval metrics:
    # Once you have a held-out query set and a business corpus, compute:
    #   NDCG@k, MRR, Recall@k  using sentence_transformers.util.semantic_search
    # or integrate with your production ES k-NN index.

    # ── Save results ─────────────────────────────────────────────────────
    results_path = output_dir / f"eval_{split}_results.json"
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    logger.info("Results saved → %s", results_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate embedding quality of the fine-tuned model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/checkpoints/best_model"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/checkpoints/best_model"),
        help="Where to write eval result JSON files",
    )
    p.add_argument("--split",       default="val", choices=["val", "test"])
    p.add_argument("--max_samples", type=int, default=2_000)
    p.add_argument("--es_host",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    es_cfg = ESConfig()
    if args.es_host:
        es_cfg.host = args.es_host

    evaluate(
        model_dir=args.model_dir,
        es_cfg=es_cfg,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
