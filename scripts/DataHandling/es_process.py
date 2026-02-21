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
from collections import defaultdict
import logging
import os
import random
from pathlib import Path
import string

from elasticsearch import Elasticsearch
# USer scan to stream results of the restaurant search.
from elasticsearch.helpers import bulk, scan
from tqdm import tqdm
from business_dto import BusinessDTO
from templates import TEMPLATES
import hashlib

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
            "query_tier": {"type": "keyword"},
            "business_text": {"type": "text"},
            "business_id": {"type": "keyword"},
            # "train" | "val" | "test"  — used to filter in ElasticSearchDataset
            "split": {"type": "keyword"},
            # Random integer used for shuffle-buffer ordering in the DataLoader
            "shuffle_id": {"type": "integer"},
        }
    },
}

_SLOT_SOURCES = {
    "cat":      lambda dto: list(dto.categories),
    "phrase":   lambda dto: list(dto.phrases),
    "city":     lambda dto: [dto.city],
    "state":    lambda dto: [dto.state],
    "location": lambda dto: [dto.location],
    "name":     lambda dto: [dto.name],
    "amb":   lambda dto: list(dto.nested_attrs.get("Ambience", frozenset())),
    "price": lambda dto: {
        "1": ["cheap", "affordable", "budget"],
        "2": ["moderate", "fairly priced"],
        "3": ["expensive", "upscale"],
        "4": ["luxury", "high-end", "fancy"],
    }.get(dto.enum_attrs.get("RestaurantsPriceRange2", "2"), ["moderate"]),
    "noise": lambda dto: {
        "None":    ["silent"],
        "quiet":   ["quiet"],
        "average": ["normal sounding"],
        "loud":    ["loud"],
    }.get(dto.enum_attrs.get("NoiseLevel", "average"), ["normal sounding"]),
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

def build_business_text(dto: BusinessDTO) -> str:
    """Compose a natural-language description of the business.

    This text becomes the "document" side of each contrastive training pair.

    Example output
    --------------
    "Bella Italia — Italian, Pizza — romantic atmosphere,
     accepts reservations — Seattle, WA"
    """
    parts = [dto.name]
    if dto.categories:
        parts.append(", ".join(dto.categories))
    if dto.phrases:
        parts.append(", ".join(dto.phrases))
    parts.append(dto.location)
    return " — ".join(parts)

    
def _apply_templates(dto: BusinessDTO, templates: dict[str, list[str]]) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    formatter = string.Formatter()

    for tier_name, tier_templates in templates.items():
        tier_queries = []
        for template in tier_templates:
            fields = {fname for _, fname, _, _ in formatter.parse(template) if fname}
            pools = {
                prefix: list(source(dto)) for prefix, source in _SLOT_SOURCES.items()
            }
            resolved: dict[str, str] = {}
            skip = False

            for field in fields:
                if field in resolved:
                    continue
                prefix = next((p for p in _SLOT_SOURCES if field == p or field.startswith(p)), None)
                if prefix is None or not pools[prefix]:
                    skip = True
                    break
                val = random.choice(pools[prefix])
                pools[prefix].remove(val)
                resolved[field] = val

            if not skip:
                tier_queries.append(template.format_map(defaultdict(str, resolved)))

        if tier_queries:
            results[tier_name] = tier_queries

    return results

            


def generate_queries(dto: BusinessDTO) -> dict[str, list[str]]:
    """Return synthetic user queries grouped by template tier.

    Returns
    -------
    dict[str, list[str]]
        Tier name → list of generated queries for that tier.
    """
    return _apply_templates(dto, TEMPLATES)


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
    h = int(hashlib.md5(business_id.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"

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
                    dto = BusinessDTO.from_raw(hit["_source"])

                    biz_text = build_business_text(dto)
                    tier_queries = generate_queries(dto)
                    split = assign_split(dto.business_id)

                    for tier_name, queries in tier_queries.items():
                        for query in queries:
                            actions.append(
                                {
                                    "_index": target_index,
                                    "_source": {
                                        "query": query,
                                        "query_tier": tier_name,
                                        "business_text": biz_text,
                                        "business_id": dto.business_id,
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
