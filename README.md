# Complementary Learning System in NumPy

This repository contains a cheap, biologically grounded Complementary Learning
System (CLS) implemented with only Python and NumPy.

The model separates memory into two systems:

- `Hippocampus`: fast one-shot sparse episodic memory, inspired by dentate gyrus
  pattern separation and CA3 content-addressable recall.
- `IndexedSparseNeocortex`: slow sparse distributed cortical memory with many
  local minhash indexes. This avoids the usual global-bundle SNR collapse by
  making retrieval noise depend on local bucket occupancy (`items / buckets`),
  not total memories.
- `ComplementaryLearningSystem`: wrapper that encodes sensory vectors into SDRs,
  stores them rapidly in hippocampus, and consolidates them into neocortex.

## Why The Neocortex Has Exponential Address Space

An SDR with `n` possible cortical cells and `k` active cells has:

```text
choose(n, k)
```

possible addresses. For example, `n=4096, k=64` gives more than `10^130` possible
addresses. The implementation never materializes that space. It stores only the
active indices and a small number of local hash-table references per memory.

The SNR fix is local indexing. A naive superposition memory adds every item into
the same vector, so crosstalk grows with all stored memories. Here each item is
routed through multiple minhash tables. If `M` memories are stored and each table
has `B` buckets, expected local load is approximately `M / B`. Retrieval examines
only memories sharing buckets with the query and then reranks by exact SDR
overlap.

## Usage

```python
import numpy as np

from cls_numpy import ComplementaryLearningSystem

rng = np.random.default_rng(0)
cls = ComplementaryLearningSystem(input_dim=64, sdr_size=2048, active_bits=40)

x = rng.normal(size=64)
cls.observe(x, payload="first memory")

print(cls.infer(x).payload)  # fast hippocampal recall
cls.consolidate()
print(cls.infer(x, prefer_hippocampus=False).payload)  # cortical recall
```

Run the demo:

```bash
python cls_numpy.py
```

Run verification tests:

```bash
python -m unittest -v
```
