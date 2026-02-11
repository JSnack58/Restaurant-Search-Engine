"""
ElasticSearch-backed IterableDataset
=====================================
Streams (query, business_text) documents from the yelp_training_pairs index
using ES Point-in-Time + search_after for memory-efficient iteration.

Includes an in-memory shuffle buffer to approximate random sampling without
loading the full index into memory.

Design notes
------------
• num_workers=0 is the correct default for a local single-node ES instance.
  Every DataLoader worker opens its own ES connection and PIT session; with
  multiple workers you get duplicate data unless you add worker-sharding logic.

  ✏️ PLACEHOLDER (future): if tokenization becomes the CPU bottleneck, add
  worker-sharding:
      1. In __iter__, call torch.utils.data.get_worker_info()
      2. Add a filter on shuffle_id:
             {"range": {"shuffle_id": {"gte": worker_start, "lt": worker_end}}}
      3. Set dataloader_num_workers > 0 in TrainingConfig

• The dataset yields plain dicts {"query": str, "business_text": str}.
  es_pairs_dataset.py wraps this class to produce InputExample objects.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Iterator

from elasticsearch import Elasticsearch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class ElasticSearchDataset(IterableDataset):
    """
    Streams raw training-pair dicts from an ES index, filtered by split.

    Parameters
    ----------
    es_host       : Elasticsearch host URL
    index_name    : Name of the yelp_training_pairs index
    split         : "train", "val", or "test"
    es_batch_size : Number of documents fetched per ES search request
    shuffle_buffer: Size of the in-memory shuffle buffer (rows)
    pit_keep_alive: How long to keep the PIT alive between requests

    Yields
    ------
    dict with keys: "query" (str), "business_text" (str), "business_id" (str)
    """

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        index_name: str = "yelp_training_pairs",
        split: str = "train",
        es_batch_size: int = 256,
        shuffle_buffer: int = 5_000,
        pit_keep_alive: str = "2m",
    ) -> None:
        self.es = Elasticsearch(
            es_host
            # ✏️ PLACEHOLDER: pass api_key or basic_auth for secured clusters
        )
        self.index_name = index_name
        self.split = split
        self.es_batch_size = es_batch_size
        self.shuffle_buffer = shuffle_buffer
        self.pit_keep_alive = pit_keep_alive

    # ------------------------------------------------------------------
    # Internal: PIT streaming loop
    # ------------------------------------------------------------------
    def _stream_docs(self) -> Iterator[dict]:
        """
        Open a PIT, page through all documents matching `split` via
        search_after, and yield raw _source dicts.  Always closes the PIT
        in a finally block to avoid resource leaks on the ES server.
        """
        query = {"term": {"split": self.split}}

        pit = self.es.open_point_in_time(
            index=self.index_name,
            keep_alive=self.pit_keep_alive,
        )
        pit_id: str = pit["id"]
        search_after = None

        try:
            while True:
                body: dict = {
                    "size": self.es_batch_size,
                    "pit": {"id": pit_id, "keep_alive": self.pit_keep_alive},
                    "sort": [{"_shard_doc": "asc"}],
                    "query": query,
                }
                if search_after:
                    body["search_after"] = search_after

                resp = self.es.search(body=body)
                hits = resp["hits"]["hits"]
                if not hits:
                    break

                pit_id = resp["pit_id"]
                search_after = hits[-1]["sort"]

                for hit in hits:
                    yield hit["_source"]

        finally:
            self.es.close_point_in_time(id=pit_id)

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[dict]:
        """
        Yield shuffled training-pair dicts using a fill-and-flush buffer.

        ✏️ PLACEHOLDER — extra query filters:
        To restrict training to a subset, add a bool/filter clause to
        `query` in _stream_docs().  For example, to train only on
        restaurants in a specific city:
            query = {
                "bool": {
                    "filter": [
                        {"term": {"split": self.split}},
                        {"term": {"city": "Las Vegas"}},  # add here
                    ]
                }
            }
        """
        buffer: list[dict] = []

        for doc in self._stream_docs():
            buffer.append(doc)
            if len(buffer) >= self.shuffle_buffer:
                random.shuffle(buffer)
                yield from buffer
                buffer = []

        if buffer:
            random.shuffle(buffer)
            yield from buffer
