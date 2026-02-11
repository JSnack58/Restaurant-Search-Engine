"""
Pair / Triplet Dataset Builder
================================
Wraps ElasticSearchDataset to produce sentence_transformers.InputExample
objects in the format expected by the contrastive training loop.

MNRL mode   → InputExample(texts=[query, business_text])
Triplet mode → InputExample(texts=[query, positive_biz_text, negative_biz_text])

Glue code (complete): random pair/triplet construction within a batch buffer.

✏️ PLACEHOLDER (hard-negative mining):
The random negative in triplet mode is the lowest-effort baseline.  For a
stronger signal, replace the random negative with the *hardest* negative —
the business whose embedding is closest to the anchor query without being the
true positive.  This typically requires a pre-built FAISS or ES k-NN index of
business embeddings.  See the placeholder comment in _build_triplet().
"""
from __future__ import annotations

import random
from typing import Iterator

from sentence_transformers import InputExample
from torch.utils.data import IterableDataset

from scripts.Finetuning.es_dataset import ElasticSearchDataset


class ESPairsDataset(IterableDataset):
    """
    Yields InputExample objects constructed from yelp_training_pairs.

    Parameters
    ----------
    es_host        : Elasticsearch host URL
    index_name     : yelp_training_pairs index
    split          : "train", "val", or "test"
    mode           : "mnrl" for pairs, "triplet" for triples
    es_batch_size  : Documents fetched per ES request
    shuffle_buffer : Size of in-memory shuffle buffer
    pit_keep_alive : PIT keep-alive duration
    """

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        index_name: str = "yelp_training_pairs",
        split: str = "train",
        mode: str = "mnrl",
        es_batch_size: int = 256,
        shuffle_buffer: int = 5_000,
        pit_keep_alive: str = "2m",
    ) -> None:
        if mode not in {"mnrl", "triplet"}:
            raise ValueError(f"mode must be 'mnrl' or 'triplet', got '{mode!r}'")

        self._source = ElasticSearchDataset(
            es_host=es_host,
            index_name=index_name,
            split=split,
            es_batch_size=es_batch_size,
            shuffle_buffer=shuffle_buffer,
            pit_keep_alive=pit_keep_alive,
        )
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[InputExample]:
        """
        Collect docs into a local buffer, then emit InputExample objects.

        The buffer size controls negative diversity: a larger buffer means
        the random negative is less likely to accidentally be a semantically
        similar business.
        """
        buffer: list[dict] = []

        for doc in self._source:
            buffer.append(doc)
            if len(buffer) >= self.shuffle_buffer:
                yield from self._emit(buffer)
                buffer = []

        if buffer:
            yield from self._emit(buffer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit(self, buffer: list[dict]) -> Iterator[InputExample]:
        random.shuffle(buffer)
        if self.mode == "mnrl":
            for doc in buffer:
                yield self._build_pair(doc)
        else:
            biz_texts = [d["business_text"] for d in buffer]
            for i, doc in enumerate(buffer):
                yield self._build_triplet(doc, biz_texts, i)

    @staticmethod
    def _build_pair(doc: dict) -> InputExample:
        """
        MNRL: (query, positive_business_text).
        In-batch negatives are constructed inside compute_mnrl_loss().
        """
        return InputExample(texts=[doc["query"], doc["business_text"]])

    @staticmethod
    def _build_triplet(
        doc: dict,
        biz_texts: list[str],
        self_idx: int,
    ) -> InputExample:
        """
        Triplet: (query, positive_biz_text, negative_biz_text).

        The negative is currently a random business from the same buffer.

        ✏️ PLACEHOLDER — hard-negative mining:
        Replace the random negative with the hardest negative in the batch:
            1. Compute cosine similarity between doc["query"] and all biz_texts
               using the current model checkpoint.
            2. Sort by descending similarity.
            3. Pick the highest-similarity biz_text that is NOT self_idx.
        This requires passing the model as an argument to this method and
        calling model.encode() (with torch.no_grad()) to get candidate scores.
        """
        candidates = [t for i, t in enumerate(biz_texts) if i != self_idx]
        negative = random.choice(candidates)
        return InputExample(texts=[doc["query"], doc["business_text"], negative])
