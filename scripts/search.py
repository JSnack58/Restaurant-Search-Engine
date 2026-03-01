"""
Interactive Restaurant Search
==============================
Loads the fine-tuned embedding model once, then enters a query loop that
encodes each query and runs an Elasticsearch kNN search against the
``yelp_businesses_knn`` index.

Usage
-----
    python scripts/search.py
    make search

    # With a non-default index or model:
    python scripts/search.py --knn_index yelp_businesses_knn --k 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL     = "models/embedding_model/sentence_transformer"
_DEFAULT_KNN_INDEX = "yelp_businesses_knn"
_DEFAULT_K         = 10


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def knn_search(
    es: Elasticsearch,
    model: SentenceTransformer,
    query: str,
    knn_index: str,
    k: int,
) -> list[dict]:
    vector = model.encode(query).tolist()
    knn_clause = {
        "field":          "embedding",
        "query_vector":   vector,
        "k":              k,
        "num_candidates": k * 10,
    }
    resp = es.search(index=knn_index, body={"knn": knn_clause, "size": k})
    return resp["hits"]["hits"]


def _print_results(hits: list[dict]) -> None:
    if not hits:
        print("  (no results)\n")
        return
    for i, hit in enumerate(hits, 1):
        src   = hit["_source"]
        score = hit["_score"]
        name  = src.get("name", "?")
        city  = src.get("city", "")
        state = src.get("state", "")
        text  = src.get("business_text", "")[:120]
        print(f"  {i:>2}. {name} — {city}, {state}  [score: {score:.4f}]")
        print(f"      {text}")
    print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def run_repl(
    model_path: str,
    knn_index: str,
    es_host: str,
    k: int,
) -> None:
    print(f"Loading model from {model_path} …")
    model = SentenceTransformer(model_path)

    es_host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
    es = Elasticsearch(es_host)
    print(f"Connected to ES at {es_host}")
    print(f"Searching index: {knn_index}  |  top-k: {k}")
    print('Type a query, or "quit" / Ctrl-C to exit.\n')

    try:
        while True:
            query = input("Search: ").strip()
            if not query or query.lower() in {"quit", "exit", "q"}:
                break

            hits = knn_search(es, model, query, knn_index, k)
            _print_results(hits)

    except KeyboardInterrupt:
        pass

    print("\nGoodbye.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive kNN restaurant search against Elasticsearch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", default=_DEFAULT_MODEL,     help="SentenceTransformer path")
    p.add_argument("--knn_index",  default=_DEFAULT_KNN_INDEX,  help="ES kNN index name")
    p.add_argument("--es_host",    default=None,                help="ES host URL")
    p.add_argument("--k",          type=int, default=_DEFAULT_K, help="Number of results to return")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_repl(
        model_path=args.model_path,
        knn_index=args.knn_index,
        es_host=args.es_host,
        k=args.k,
    )
