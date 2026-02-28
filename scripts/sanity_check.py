"""
Sanity Check — Query → Top-3 Businesses
=========================================
Encodes a list of queries and a sample of business texts from ES,
then prints the top-3 closest businesses per query using cosine similarity.

Usage
-----
    # Against the current best_model checkpoint:
    python scripts/sanity_check.py

    # Against a specific checkpoint:
    python scripts/sanity_check.py --model_dir models/checkpoints/checkpoint-epoch-1

    # Against the pretrained (unfine-tuned) base model:
    python scripts/sanity_check.py --model_dir bert-base-uncased
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.Finetuning.config import ESConfig
from scripts.Finetuning.es_dataset import ElasticSearchDataset

QUERIES = [
    "cheap sushi in Phoenix",
    "romantic Italian dinner",
    "late night bbq near me",
    "best brunch spots downtown",
    "vegan friendly cafe",
    "loud sports bar with wings",
]

def run(model_dir: str, es_cfg: ESConfig, corpus_size: int) -> None:
    print(f"Loading model: {model_dir}")
    model = SentenceTransformer(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # ── Stream a corpus of business texts from ES ─────────────────────────
    print(f"Streaming {corpus_size} business texts from ES…")
    dataset = ElasticSearchDataset(
        es_host=es_cfg.host,
        index_name=es_cfg.training_pairs_index,
        split="test",
        es_batch_size=256,
        shuffle_buffer=corpus_size,
        max_samples=corpus_size,
    )

    business_texts: list[str] = []
    seen: set[str] = set()
    for doc in dataset:
        bt = doc["business_text"]
        if bt not in seen:
            seen.add(bt)
            business_texts.append(bt)

    print(f"Unique business texts: {len(business_texts)}\n")

    # ── Encode corpus and queries ─────────────────────────────────────────
    with torch.no_grad():
        corpus_emb = model.encode(
            business_texts, convert_to_tensor=True, device=device, show_progress_bar=True
        )
        corpus_emb = F.normalize(corpus_emb, dim=-1)

        query_emb = model.encode(
            QUERIES, convert_to_tensor=True, device=device
        )
        query_emb = F.normalize(query_emb, dim=-1)

    # ── Cosine similarity: (num_queries, corpus_size) ─────────────────────
    scores = query_emb @ corpus_emb.T

    # ── Print top-3 per query ─────────────────────────────────────────────
    for i, query in enumerate(QUERIES):
        top_scores, top_idx = scores[i].topk(3)
        print(f"Query: {query!r}")
        for rank, (idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), 1):
            print(f"  {rank}. [{score:.3f}] {business_texts[idx]}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        default="models/checkpoints/best_model",
        help="SentenceTransformer checkpoint path or HF model name",
    )
    p.add_argument(
        "--corpus_size",
        type=int,
        default=5_000,
        help="Number of business texts to load from ES as the search corpus",
    )
    args = p.parse_args()

    es_cfg = ESConfig()
    run(args.model_dir, es_cfg, args.corpus_size)
