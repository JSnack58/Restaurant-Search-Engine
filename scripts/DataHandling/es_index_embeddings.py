"""
ES kNN Embedding Indexer
========================
Encodes all restaurant businesses using the fine-tuned SentenceTransformer and
writes them into an Elasticsearch index with a ``dense_vector`` field for
approximate nearest-neighbour (kNN) search.

Run this once after training (and after ``make export``) to build the search index.

Usage
-----
    python scripts/DataHandling/es_index_embeddings.py
    make index_embeddings

    # Rebuild from scratch (drops existing index):
    python scripts/DataHandling/es_index_embeddings.py --recreate
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from business_dto import BusinessDTO
from es_process import build_business_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL        = "models/embedding_model/sentence_transformer"
_DEFAULT_SOURCE_INDEX = "yelp_businesses_raw"
_DEFAULT_KNN_INDEX    = "yelp_businesses_knn"
_DEFAULT_BATCH        = 128


# ---------------------------------------------------------------------------
# ES helpers
# ---------------------------------------------------------------------------
def _get_es(host: str) -> Elasticsearch:
    return Elasticsearch(host or os.getenv("ES_HOST", "http://localhost:9200"))


def _create_knn_index(es: Elasticsearch, index: str, dims: int) -> None:
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "business_id":   {"type": "keyword"},
                "name":          {"type": "text"},
                "city":          {"type": "keyword"},
                "state":         {"type": "keyword"},
                "categories":    {"type": "keyword"},
                "business_text": {"type": "text"},
                "embedding": {
                    "type":       "dense_vector",
                    "dims":       dims,
                    "index":      True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es.indices.create(index=index, body=mapping)
    logger.info("Created kNN index: %s (dims=%d)", index, dims)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def index_embeddings(
    model_path: str,
    source_index: str,
    knn_index: str,
    es_host: str,
    encode_batch: int,
    recreate: bool,
) -> None:
    es = _get_es(es_host)

    logger.info("Loading model from %s", model_path)
    model = SentenceTransformer(model_path)
    dims = model.get_sentence_embedding_dimension()
    logger.info("Embedding dims: %d", dims)

    if es.indices.exists(index=knn_index):
        if recreate:
            es.indices.delete(index=knn_index)
            logger.info("Dropped existing index: %s", knn_index)
            _create_knn_index(es, knn_index, dims)
        else:
            logger.info("Index already exists (use --recreate to rebuild): %s", knn_index)
    else:
        _create_knn_index(es, knn_index, dims)

    # ── Stream + encode + bulk-index ─────────────────────────────────────
    buf_texts: list[str] = []
    buf_docs:  list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal total
        if not buf_docs:
            return
        embeddings = model.encode(buf_texts, batch_size=encode_batch, show_progress_bar=False)
        actions = [
            {
                "_index": knn_index,
                "_id":    doc["business_id"],
                "_source": {**doc, "embedding": emb.tolist()},
            }
            for doc, emb in zip(buf_docs, embeddings)
        ]
        bulk(es, actions)
        total += len(actions)
        buf_texts.clear()
        buf_docs.clear()

    logger.info("Streaming from '%s' — this may take a while…", source_index)
    hits = scan(es, index=source_index, query={"query": {"match_all": {}}}, scroll="5m")

    for hit in tqdm(hits, desc="Encoding & indexing"):
        src = hit["_source"]
        try:
            dto = BusinessDTO.from_raw(src)
        except Exception:
            continue
        if not dto.categories:
            continue

        buf_texts.append(build_business_text(dto))
        buf_docs.append({
            "business_id": dto.business_id,
            "name":        dto.name,
            "city":        dto.city,
            "state":       dto.state,
            "categories":  list(dto.categories),
            "business_text": build_business_text(dto),
        })

        if len(buf_docs) >= encode_batch:
            flush()

    flush()
    logger.info("Done. Indexed %d restaurants into '%s'.", total, knn_index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed all businesses and index them into an ES kNN index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path",    default=_DEFAULT_MODEL,        help="SentenceTransformer path")
    p.add_argument("--source_index",  default=_DEFAULT_SOURCE_INDEX,  help="ES index to read businesses from")
    p.add_argument("--knn_index",     default=_DEFAULT_KNN_INDEX,     help="ES index to write embeddings to")
    p.add_argument("--es_host",       default=None,                   help="ES host URL")
    p.add_argument("--encode_batch",  type=int, default=_DEFAULT_BATCH, help="Encoding batch size")
    p.add_argument("--recreate",      action="store_true",            help="Drop and rebuild the kNN index")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    index_embeddings(
        model_path=args.model_path,
        source_index=args.source_index,
        knn_index=args.knn_index,
        es_host=args.es_host,
        encode_batch=args.encode_batch,
        recreate=args.recreate,
    )
