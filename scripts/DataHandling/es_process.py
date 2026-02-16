"""
Business Processing Pipeline
============================
Reads from yelp_businesses_raw, calls team-defined stubs to generate
(query, business_text) training pairs, assigns train/val/test splits by
business_id, then bulk-indexes into yelp_training_pairs.

Glue code (complete):  index creation, PIT streaming, bulk write, argparse CLI.
Team stubs:
    build_business_text()  — compose a text description of the business
    generate_queries()     — produce synthetic user queries
    assign_split()         — deterministic split assignment by business_id

Usage
-----
    python scripts/DataHandling/es_process.py \\
        --source_index yelp_businesses_raw \\
        --target_index yelp_training_pairs
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

from elasticsearch import Elasticsearch
# USer scan to stream results of the restaurant search.
from elasticsearch.helpers import bulk, scan
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index mapping
# ---------------------------------------------------------------------------
_TRAINING_INDEX_MAPPING = {
    "settings": {
        # ✏️ PLACEHOLDER: tune for your cluster.
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
    },
    "mappings": {
        "properties": {
            "query": {"type": "text"},
            "business_text": {"type": "text"},
            "business_id": {"type": "keyword"},
            # "train" | "val" | "test"  — used to filter in ElasticSearchDataset
            "split": {"type": "keyword"},
            # Random integer used for shuffle-buffer ordering in the DataLoader
            "shuffle_id": {"type": "integer"},
        }
    },
}


# ---------------------------------------------------------------------------
# ES client factory (mirrors es_ingest_businesses.py)
# ---------------------------------------------------------------------------
def get_es_client(host: str | None = None) -> Elasticsearch:
    """
    ✏️ PLACEHOLDER: add auth for secured clusters (see es_ingest_businesses.py).
    Default: unauthenticated local Docker instance.
    """
    resolved = host or os.getenv("ES_HOST", "http://localhost:9200")
    return Elasticsearch(resolved)


def create_training_index(es: Elasticsearch, index_name: str) -> None:
    if es.indices.exists(index=index_name):
        logger.info("Index already exists, skipping creation: %s", index_name)
        return
    es.indices.create(index=index_name, body=_TRAINING_INDEX_MAPPING)
    logger.info("Created index: %s", index_name)


# ---------------------------------------------------------------------------
# ✏️  TEAM STUBS — implement the three functions below
# ---------------------------------------------------------------------------

def build_business_text(business: dict) -> str:
    """
    ✏️ PLACEHOLDER — Compose a single natural-language text string that
    describes the business.  This text becomes the "document" side of each
    contrastive training pair.

    The string should capture the key signals a user query might match:
    name, cuisine/category, relevant attributes, and location.

    Example outputs
    ---------------
    "Bella Italia — Italian, Pizza, Dine-In — romantic atmosphere,
     accepts reservations — Seattle, WA"

    "Joe's Burger Joint — Fast Food, Burgers, American — casual,
     takeout available, kid-friendly — Los Angeles, CA"

    Parameters
    ----------
    business : dict
        A single document from yelp_businesses_raw (_source).
        Relevant keys: name, categories, attributes, city, state.

    Returns
    -------
    str
        A natural-language description of the business.

    Raises
    ------
    NotImplementedError
    """
    return f"""{business["name"]} - {business['categories']} - {business['attributes']} - {business['city']},{business['state']}"""

def _broad_queries(business: dict) -> list[str]:
    broad_queries = []
    # Names, Categories, attribute
    broad_queries.extend([business['name']] + business['categories'])
    broad_queries.extend([attr for attr in business['attributes'] if business['attributes'][attr] == "True"])

    return broad_queries


def generate_queries(business: dict) -> list[str]:
    """
    ✏️ PLACEHOLDER — Return a list of synthetic user queries for this business.

    Queries should be phrased as a user would type them into a restaurant
    search engine, for example:
        "I'm looking for authentic Italian pasta"
        "romantic restaurant in Seattle"
        "cheap family-friendly burgers near me"
        "where can I get good pizza for takeout?"

    Strategy options (team decides)
    ---------------------------------
    A) Template-based:
       Map category/attribute combinations to query slot-fill templates.
       E.g., if categories contains "Italian" and attributes["Ambience"]["romantic"]:
           queries.append("romantic Italian restaurant")

    B) LLM-assisted:
       Call an API (OpenAI, Anthropic Claude, etc.) with the business details
       and ask it to generate N realistic user queries.

    C) Rule-based expansion:
       Define attribute → query phrase dictionaries and combine them.

    Parameters
    ----------
    business : dict
        A single document from yelp_businesses_raw (_source).

    Returns
    -------
    list[str]
        One or more natural-language queries relevant to this business.
        Returning multiple queries per business multiplies training data size.

    Raises
    ------
    NotImplementedError
    """
    queries = []
    # Vague
    # Rich vibe description
    # Rich service description


def assign_split(business_id: str) -> str:
    """
    ✏️ PLACEHOLDER — Return "train", "val", or "test" for a given business.

    IMPORTANT: This function MUST be deterministic and partition by business_id,
    not by row index.  All queries generated for a given business must land in
    the same split to prevent data leakage between splits.

    Suggested implementation (deterministic hash, 80 / 10 / 10 split):
    -------------------------------------------------------------------
        import hashlib
        h = int(hashlib.md5(business_id.encode()).hexdigest(), 16) % 100
        if h < 80:
            return "train"
        elif h < 90:
            return "val"
        else:
            return "test"

    Parameters
    ----------
    business_id : str
        The Yelp business_id string (unique per business).

    Returns
    -------
    str
        One of "train", "val", or "test".

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Implement assign_split() in scripts/DataHandling/es_process.py"
    )


# ---------------------------------------------------------------------------
# Main processing loop (glue — complete)
# ---------------------------------------------------------------------------
def process_businesses(
    es: Elasticsearch,
    source_index: str,
    target_index: str,
    batch_size: int = 500,
    pit_keep_alive: str = "2m",
) -> None:
    """
    Stream every business from source_index, generate training pairs, and
    bulk-index them into target_index.

    Uses ES Point-in-Time + search_after for memory-efficient streaming of
    the full index regardless of size.
    """
    create_training_index(es, target_index)

    pit = es.open_point_in_time(index=source_index, keep_alive=pit_keep_alive)
    pit_id = pit["id"]
    search_after = None
    total_pairs = 0

    try:
        with tqdm(desc="Processing businesses") as pbar:
            while True:
                body: dict = {
                    "size": batch_size,
                    "pit": {"id": pit_id, "keep_alive": pit_keep_alive},
                    "sort": [{"_shard_doc": "asc"}],
                }
                if search_after:
                    body["search_after"] = search_after

                resp = es.search(body=body)
                hits = resp["hits"]["hits"]
                if not hits:
                    break

                pit_id = resp["pit_id"]
                search_after = hits[-1]["sort"]

                actions: list[dict] = []
                for hit in hits:
                    business = hit["_source"]
                    biz_id: str = business.get("business_id", hit["_id"])

                    biz_text = build_business_text(business)
                    queries = generate_queries(business)
                    split = assign_split(biz_id)

                    for query in queries:
                        actions.append(
                            {
                                "_index": target_index,
                                "_source": {
                                    "query": query,
                                    "business_text": biz_text,
                                    "business_id": biz_id,
                                    "split": split,
                                    "shuffle_id": random.randint(0, 999_999),
                                },
                            }
                        )

                if actions:
                    bulk(es, actions)
                    total_pairs += len(actions)

                pbar.update(len(hits))

    finally:
        es.close_point_in_time(id=pit_id)

    logger.info("Done. Indexed %d training pairs → %s", total_pairs, target_index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate (query, business_text) training pairs and index into ES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source_index", default="yelp_businesses_raw")
    p.add_argument("--target_index", default="yelp_training_pairs")
    p.add_argument("--es_host", default=None)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--pit_keep_alive", default="2m")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    es = get_es_client(args.es_host)
    process_businesses(
        es=es,
        source_index=args.source_index,
        target_index=args.target_index,
        batch_size=args.batch_size,
        pit_keep_alive=args.pit_keep_alive,
    )
