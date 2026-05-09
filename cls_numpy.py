"""Cheap NumPy Complementary Learning System.

The model has two memories with different biological roles:

* Hippocampus: fast one-shot sparse episodic indexing, like DG/CA3.
* Neocortex: slow indexed sparse distributed representations (SDRs), like
  cortical columns using local competition and sparse synaptic addressing.

The neocortical memory does not put every item into one global superposition.
Each SDR is inserted into many local minhash tables. Retrieval only compares the
small set of memories sharing local hash buckets with the query, so the noise is
set by bucket occupancy rather than total memory size. The representational
address space is combinatorial: choose(sdr_size, active_bits).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log10, sqrt
from typing import Any, Iterable

import numpy as np


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k > x.size:
        raise ValueError("k must be in [1, x.size]")
    idx = np.argpartition(x, -k)[-k:]
    return np.sort(idx.astype(np.int32, copy=False))


def _normalize_sdr(sdr: Iterable[int], sdr_size: int, active_bits: int | None) -> np.ndarray:
    out = np.asarray(list(sdr), dtype=np.int32)
    if out.ndim != 1:
        raise ValueError("SDR must be a 1D iterable of active bit indices")
    if out.size == 0:
        raise ValueError("SDR cannot be empty")
    out = np.unique(out)
    if out[0] < 0 or out[-1] >= sdr_size:
        raise ValueError("SDR index out of range")
    if active_bits is not None and out.size != active_bits:
        raise ValueError(f"expected {active_bits} active bits, got {out.size}")
    return out.astype(np.int32, copy=False)


def sdr_overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Return the number of shared active bits in two sorted SDR index arrays."""

    return int(np.intersect1d(a, b, assume_unique=True).size)


def corrupt_sdr(
    sdr: Iterable[int],
    sdr_size: int,
    drop_fraction: float = 0.15,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return an SDR with some active bits replaced by random inactive bits."""

    rng = _as_rng(seed)
    clean = _normalize_sdr(sdr, sdr_size, active_bits=None)
    n_drop = int(round(clean.size * drop_fraction))
    if n_drop == 0:
        return clean.copy()

    keep = np.ones(clean.size, dtype=bool)
    keep[rng.choice(clean.size, size=n_drop, replace=False)] = False
    kept = clean[keep]
    inactive = np.setdiff1d(np.arange(sdr_size, dtype=np.int32), clean, assume_unique=True)
    added = rng.choice(inactive, size=n_drop, replace=False).astype(np.int32)
    return np.sort(np.concatenate([kept, added])).astype(np.int32)


def splitmix64(values: np.ndarray | np.uint64, salt: np.uint64 = np.uint64(0)) -> np.ndarray:
    """Deterministic 64-bit mixing for stable SDR hashing."""

    x = np.asarray(values, dtype=np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x &= UINT64_MASK
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x &= UINT64_MASK
    return x ^ (x >> np.uint64(31))


@dataclass(frozen=True)
class QueryResult:
    payload: Any | None
    item_id: int | None
    score: float
    overlap: float
    bucket_matches: int
    candidates_examined: int


@dataclass(frozen=True)
class Episode:
    sdr: np.ndarray
    payload: Any


class SparseRandomProjector:
    """Sparse random projection encoder for sensory vectors.

    This approximates expansion into a high-dimensional sparse code, similar to
    dentate/cortical sparse activation. It stores only fan-in indices and signs,
    not a dense projection matrix.
    """

    def __init__(
        self,
        input_dim: int,
        sdr_size: int = 4096,
        active_bits: int = 64,
        fan_in: int = 32,
        seed: int | np.random.Generator | None = 0,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 < active_bits < sdr_size:
            raise ValueError("active_bits must be in (0, sdr_size)")
        if fan_in <= 0:
            raise ValueError("fan_in must be positive")
        self.input_dim = int(input_dim)
        self.sdr_size = int(sdr_size)
        self.active_bits = int(active_bits)
        self.fan_in = int(fan_in)

        rng = _as_rng(seed)
        self.indices = rng.integers(0, input_dim, size=(sdr_size, fan_in), dtype=np.int32)
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(sdr_size, fan_in))

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.input_dim,):
            raise ValueError(f"expected vector shape {(self.input_dim,)}, got {x.shape}")
        scores = (x[self.indices] * self.signs).sum(axis=1)
        return _top_k_indices(scores, self.active_bits)


class Hippocampus:
    """Fast sparse episodic memory with CA3-like content addressing."""

    def __init__(self, sdr_size: int, active_bits: int, recall_threshold: float = 0.55) -> None:
        self.sdr_size = int(sdr_size)
        self.active_bits = int(active_bits)
        self.recall_threshold = float(recall_threshold)
        self.episodes: list[Episode] = []

    def store(self, sdr: Iterable[int], payload: Any) -> int:
        key = _normalize_sdr(sdr, self.sdr_size, self.active_bits)
        self.episodes.append(Episode(key, payload))
        return len(self.episodes) - 1

    def recall(self, sdr: Iterable[int]) -> QueryResult:
        key = _normalize_sdr(sdr, self.sdr_size, self.active_bits)
        best_id: int | None = None
        best_overlap = -1
        for item_id, episode in enumerate(self.episodes):
            overlap = sdr_overlap(key, episode.sdr)
            if overlap > best_overlap:
                best_id = item_id
                best_overlap = overlap

        if best_id is None:
            return QueryResult(None, None, 0.0, 0.0, 0, 0)

        overlap_fraction = best_overlap / self.active_bits
        payload = self.episodes[best_id].payload if overlap_fraction >= self.recall_threshold else None
        return QueryResult(payload, best_id if payload is not None else None, overlap_fraction, overlap_fraction, 0, len(self.episodes))

    def replay(self, newest_first: bool = False) -> Iterable[Episode]:
        return reversed(self.episodes) if newest_first else iter(self.episodes)


class IndexedSparseNeocortex:
    """Slow cortical SDR memory with local minhash indexing.

    A naive SDR bundle eventually loses signal because every stored item adds
    crosstalk to the same synapses. Here, each cortical item is routed through
    multiple minhash tables. With ``bucket_count=B`` and ``M`` memories, expected
    local load is ``M / B`` per table; retrieval SNR is governed by that local
    load and the number of matching tables, not by all M memories at once.
    """

    def __init__(
        self,
        sdr_size: int = 4096,
        active_bits: int = 64,
        tables: int = 24,
        bucket_count: int = 65536,
        recall_threshold: float = 0.35,
        overlap_weight: float = 0.35,
        seed: int | np.random.Generator | None = 1,
    ) -> None:
        if not 0 < active_bits < sdr_size:
            raise ValueError("active_bits must be in (0, sdr_size)")
        if tables <= 0 or bucket_count <= 0:
            raise ValueError("tables and bucket_count must be positive")
        self.sdr_size = int(sdr_size)
        self.active_bits = int(active_bits)
        self.tables = int(tables)
        self.bucket_count = int(bucket_count)
        self.recall_threshold = float(recall_threshold)
        self.overlap_weight = float(overlap_weight)

        rng = _as_rng(seed)
        self.salts = rng.integers(0, np.iinfo(np.uint64).max, size=tables, dtype=np.uint64)
        self.sdrs: list[np.ndarray] = []
        self.payloads: list[Any] = []
        self.buckets: list[dict[int, list[int]]] = [dict() for _ in range(tables)]

    @property
    def item_count(self) -> int:
        return len(self.sdrs)

    def address_space_log10(self) -> float:
        """Base-10 log of the number of possible SDR addresses."""

        return log10(comb(self.sdr_size, self.active_bits))

    def expected_bucket_load(self) -> float:
        return self.item_count / self.bucket_count

    def capacity_for_bucket_load(self, mean_load: float = 2.0) -> int:
        """Items storable before average local bucket load reaches mean_load."""

        if mean_load <= 0:
            raise ValueError("mean_load must be positive")
        return int(mean_load * self.bucket_count)

    def _signatures(self, sdr: np.ndarray) -> np.ndarray:
        active = sdr.astype(np.uint64, copy=False)
        sigs = np.empty(self.tables, dtype=np.int64)
        for table, salt in enumerate(self.salts):
            mixed = splitmix64(active, salt)
            # Minhash is stable under partial cue overlap: a table still matches
            # when its minimum hash came from an active bit preserved in the cue.
            sigs[table] = int(mixed.min() % np.uint64(self.bucket_count))
        return sigs

    def store(self, sdr: Iterable[int], payload: Any) -> int:
        key = _normalize_sdr(sdr, self.sdr_size, self.active_bits)
        item_id = len(self.sdrs)
        self.sdrs.append(key)
        self.payloads.append(payload)
        for table, bucket in enumerate(self._signatures(key)):
            self.buckets[table].setdefault(int(bucket), []).append(item_id)
        return item_id

    def query(self, sdr: Iterable[int], top_k: int = 1) -> QueryResult | list[QueryResult]:
        key = _normalize_sdr(sdr, self.sdr_size, self.active_bits)
        if not self.sdrs:
            empty = QueryResult(None, None, 0.0, 0.0, 0, 0)
            return empty if top_k == 1 else [empty]

        candidate_score: dict[int, float] = {}
        candidate_matches: dict[int, int] = {}
        for table, bucket in enumerate(self._signatures(key)):
            ids = self.buckets[table].get(int(bucket), [])
            if not ids:
                continue
            weight = 1.0 / sqrt(len(ids))
            for item_id in ids:
                candidate_score[item_id] = candidate_score.get(item_id, 0.0) + weight
                candidate_matches[item_id] = candidate_matches.get(item_id, 0) + 1

        if not candidate_score:
            miss = QueryResult(None, None, 0.0, 0.0, 0, 0)
            return miss if top_k == 1 else [miss]

        ranked: list[QueryResult] = []
        for item_id, bucket_score in candidate_score.items():
            overlap = sdr_overlap(key, self.sdrs[item_id]) / self.active_bits
            score = (bucket_score / self.tables) + (self.overlap_weight * overlap)
            payload = self.payloads[item_id] if score >= self.recall_threshold else None
            ranked.append(
                QueryResult(
                    payload=payload,
                    item_id=item_id if payload is not None else None,
                    score=score,
                    overlap=overlap,
                    bucket_matches=candidate_matches[item_id],
                    candidates_examined=len(candidate_score),
                )
            )
        ranked.sort(key=lambda r: (r.score, r.overlap, r.bucket_matches), reverse=True)
        return ranked[0] if top_k == 1 else ranked[:top_k]


class ComplementaryLearningSystem:
    """Hippocampal fast memory plus slow neocortical consolidation."""

    def __init__(
        self,
        input_dim: int,
        sdr_size: int = 4096,
        active_bits: int = 64,
        seed: int = 0,
    ) -> None:
        self.encoder = SparseRandomProjector(input_dim, sdr_size, active_bits, seed=seed)
        self.hippocampus = Hippocampus(sdr_size, active_bits)
        self.neocortex = IndexedSparseNeocortex(sdr_size, active_bits, seed=seed + 1)
        self._consolidated = 0

    def observe(self, x: np.ndarray, payload: Any, consolidate: bool = False) -> int:
        sdr = self.encoder.encode(x)
        episode_id = self.hippocampus.store(sdr, payload)
        if consolidate:
            self.neocortex.store(sdr, payload)
            self._consolidated += 1
        return episode_id

    def consolidate(self, max_items: int | None = None, newest_first: bool = False) -> int:
        episodes = list(self.hippocampus.replay(newest_first=newest_first))
        pending = episodes[self._consolidated :]
        if max_items is not None:
            pending = pending[:max_items]
        for episode in pending:
            self.neocortex.store(episode.sdr, episode.payload)
        self._consolidated += len(pending)
        return len(pending)

    def infer(self, x: np.ndarray, prefer_hippocampus: bool = True) -> QueryResult:
        sdr = self.encoder.encode(x)
        if prefer_hippocampus:
            episodic = self.hippocampus.recall(sdr)
            if episodic.payload is not None:
                return episodic
        return self.neocortex.query(sdr)


def demo() -> None:
    rng = np.random.default_rng(7)
    cls = ComplementaryLearningSystem(input_dim=64, sdr_size=4096, active_bits=64, seed=7)
    for i in range(200):
        cls.observe(rng.normal(size=64), payload=f"memory-{i}")
    cls.consolidate()
    print(f"stored={cls.neocortex.item_count}")
    print(f"log10_sdr_address_space={cls.neocortex.address_space_log10():.1f}")
    print(f"expected_bucket_load={cls.neocortex.expected_bucket_load():.4f}")


if __name__ == "__main__":
    demo()
