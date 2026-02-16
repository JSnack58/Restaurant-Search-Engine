"""
Business Data Ingestion
=======================
Streams yelp_academic_dataset_business.json line-by-line and bulk-indexes
documents into the ElasticSearch "yelp_businesses_raw" index.

Glue code (complete):  ES client setup, index creation, bulk helper, argparse CLI.
Team stub:             stream_and_index() — implement the JSON reader & field extraction.

Usage
-----
    python scripts/DataHandling/es_ingest_businesses.py \\
        --input_path data/raw/yelp_academic_dataset_business.json

Environment variables (override via CLI or shell):
    ES_HOST      ElasticSearch host URL   (default: http://localhost:9200)
    ES_USER      Username for auth        (optional)
    ES_PASSWORD  Password for auth        (optional)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import json
from tqdm import tqdm

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index mapping
# ---------------------------------------------------------------------------
_INDEX_MAPPING = {
    "settings": {
        # ✏️ PLACEHOLDER: tune shards/replicas for your cluster size.
        # Single-node Docker default: 1 shard, 0 replicas.
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",  # reduce for write-heavy ingest
    },
    "mappings": {
        "properties": {
            "business_id": {"type": "keyword"},
            "name": {"type": "text"},
            "categories": {"type": "text"},
            # Store the raw attributes object without indexing every sub-field.
            # es_process.py accesses them at processing time.
            "attributes": {"type": "object", "enabled": False},
            "city": {"type": "keyword"},
            "state": {"type": "keyword"},
            "address":{"type":"text"},
            "longitude":{"type":"float"},
            "latitude":{"type":"float"}
        }
    },
}


# ---------------------------------------------------------------------------
# ES client factory
# ---------------------------------------------------------------------------
def get_es_client(host: str | None = None) -> Elasticsearch:
    """
    Build an Elasticsearch client pointing at the local self-hosted instance.

    Default: unauthenticated connection to http://localhost:9200 (Docker Compose).

    ✏️ PLACEHOLDER: uncomment the appropriate block below for a secured cluster:

        # API-key auth (Elastic Cloud or secured self-hosted):
        # return Elasticsearch(host, api_key=os.getenv("ES_API_KEY"))

        # Basic auth:
        # return Elasticsearch(
        #     host,
        #     basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD"))
        # )
    """
    resolved = host or os.getenv("ES_HOST", "http://localhost:9200")
    return Elasticsearch(resolved)


# ---------------------------------------------------------------------------
# Index management (glue — complete)
# ---------------------------------------------------------------------------
def create_businesses_index(es: Elasticsearch, index_name: str) -> None:
    """Create the raw businesses index if it does not already exist."""
    if es.indices.exists(index=index_name):
        logger.info("Index already exists, skipping creation: %s", index_name)
        return
    es.indices.create(index=index_name, body=_INDEX_MAPPING)
    logger.info("Created index: %s", index_name)


# ---------------------------------------------------------------------------
# ✏️  TEAM STUB — implement the streaming reader
# ---------------------------------------------------------------------------
def stream_and_index(
    filepath: Path,
    es: Elasticsearch,
    index_name: str,
    chunk_size: int = 500,
) -> None:
    """
    ✏️ PLACEHOLDER — Stream the Yelp business JSON file and bulk-index into ES.

    The file is newline-delimited JSON (one JSON object per line).
    Each document should expose at minimum:
        business_id  (str)      — unique Yelp ID
        name         (str)      — business display name
        categories   (str|None) — comma-separated category string, e.g. "Italian, Pizza"
        attributes   (dict|None)— nested attribute object (hours, Wi-Fi, etc.)
        city         (str)
        state        (str)

    Implementation hints
    --------------------
    1.  Open `filepath` for reading (UTF-8).
    2.  For each line, call `json.loads(line)` to get a dict.
    3.  Build an action dict:
            {"_index": index_name, "_id": doc["business_id"], "_source": doc}
    4.  Accumulate actions into a list; when the list reaches `chunk_size`,
        call `bulk(es, actions)` and reset the list.
    5.  After the loop, flush any remaining actions.
    6.  Wrap the file loop in tqdm for progress reporting.
    7.  Log total documents indexed on completion.

    Parameters
    ----------
    filepath   : Path to yelp_academic_dataset_business.json
    es         : Elasticsearch client (already connected)
    index_name : Name of the target index
    chunk_size : Number of documents per bulk request

    Raises
    ------
    NotImplementedError
    """
    class Restaurant:
        def __init__(self, business_id, name, categories, address, city,
                     state, latitude, longitude, attributes, **_extra):
            self.business_id = business_id
            self.name = name
            self.categories = categories.split(',') if categories else []
            self.address = address
            self.city = city
            self.state = state
            self.latitude = latitude
            self.longitude = longitude
            self.attributes = attributes or {}

    total = 0
    with (  open(filepath, "r", encoding="utf-8") as file):
        chunk = []
        for line in tqdm(file,desc="Indexing"):
            data = json.loads(line)
            restaurant = Restaurant(**data)
            chunk.append({
                "_index": index_name,
                "_id": restaurant.business_id,
                "_source": restaurant.__dict__,
            })
            if len(chunk) == chunk_size:
                bulk(es, chunk)
                total += len(chunk)
                chunk = []
        if chunk:
            bulk(es, chunk)
            total += len(chunk)
    logger.info("Indexed %d documents into %s", total, index_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Yelp business JSON into ElasticSearch (yelp_businesses_raw).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to yelp_academic_dataset_business.json",
    )
    p.add_argument(
        "--index_name",
        default="yelp_businesses_raw",
        help="Target ES index name",
    )
    p.add_argument(
        "--es_host",
        default=None,
        help="ES host URL (overrides ES_HOST env var)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Documents per bulk request",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    es = get_es_client(args.es_host)
    create_businesses_index(es, args.index_name)
    stream_and_index(
        filepath=args.input_path,
        es=es,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
    )
