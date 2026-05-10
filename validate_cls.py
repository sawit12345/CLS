"""Mechanism-level validation for the NumPy CLS implementation.

These checks are not a claim of full biological realism. They are deterministic
computational probes for the mechanisms the model claims to implement: dentate
decorrelation, CA3 completion, one-shot hippocampal recall, novelty modulation,
and cortical improvement from replay.
"""

from __future__ import annotations

import numpy as np

from cls_numpy import CA3AutoassociativeMemory, ComplementaryLearningSystem, unit_rows


def mean_pairwise_similarity(x: np.ndarray) -> float:
    x = unit_rows(x)
    sim = x @ x.T
    mask = ~np.eye(x.shape[0], dtype=bool)
    return float(sim[mask].mean())


def validate_dg_decorrelation() -> dict[str, float]:
    rng = np.random.default_rng(100)
    base = unit_rows(rng.normal(size=(64, 12)))
    cues = unit_rows(base + 0.18 * rng.normal(size=base.shape))
    contexts = np.zeros((64, 6))
    contexts[:, 0] = 1.0
    model = ComplementaryLearningSystem(cue_dim=12, target_dim=4, context_dim=6, dg_dim=512, rng_seed=10)
    raw_similarity = mean_pairwise_similarity(model.ec_state(cues, contexts))
    dg_codes = model.dg.encode(model.ec_state(cues, contexts), novelty=1.0)
    dg_similarity = mean_pairwise_similarity(dg_codes)
    assert dg_similarity < raw_similarity * 0.55, (raw_similarity, dg_similarity)
    return {"raw_similarity": raw_similarity, "dg_similarity": dg_similarity}


def validate_ca3_completion() -> dict[str, float]:
    rng = np.random.default_rng(101)
    ca3 = CA3AutoassociativeMemory(dim=256, sparsity=0.05, decay=1.0)
    patterns = np.zeros((20, 256))
    for row in range(patterns.shape[0]):
        idx = rng.choice(patterns.shape[1], size=ca3.k, replace=False)
        patterns[row, idx] = 1.0 / np.sqrt(ca3.k)
    for pattern in patterns:
        ca3.store(pattern.reshape(1, -1), strength=1.0)

    cue = patterns[3].copy()
    active = np.flatnonzero(cue)
    cue[rng.choice(active, size=active.size // 2, replace=False)] = 0.0
    completed = ca3.complete(cue.reshape(1, -1), steps=5)[0]
    partial_overlap = float(cue @ patterns[3])
    completed_overlap = float(completed @ patterns[3])
    nearest = int(np.argmax(completed @ patterns.T))
    assert nearest == 3, nearest
    assert completed_overlap > partial_overlap, (partial_overlap, completed_overlap)
    return {"partial_overlap": partial_overlap, "completed_overlap": completed_overlap}


def validate_one_shot_recall_and_novelty() -> dict[str, float]:
    rng = np.random.default_rng(102)
    model = ComplementaryLearningSystem(cue_dim=10, target_dim=5, context_dim=5, dg_dim=512, rng_seed=12)
    cues = unit_rows(rng.normal(size=(5, 10)))
    contexts = np.eye(5)
    labels = np.arange(5)
    novelty = []
    for cue, context, label in zip(cues, contexts, labels):
        novelty.append(float(model.learn_episode(cue, int(label), context, replay_steps=2)[0]))
    accuracy = float((model.classify(cues, contexts) == labels).mean())
    repeated_novelty = float(model.novelty(cues[:1], contexts[:1], np.eye(5)[0:1])[0])
    assert accuracy == 1.0, accuracy
    assert repeated_novelty < novelty[0], (novelty[0], repeated_novelty)
    return {"one_shot_accuracy": accuracy, "first_novelty": novelty[0], "repeated_novelty": repeated_novelty}


def validate_cortical_replay_improvement() -> dict[str, float]:
    rng = np.random.default_rng(103)
    centers = unit_rows(rng.normal(size=(4, 14)))
    cues = np.vstack([unit_rows(center + 0.15 * rng.normal(size=(12, 14))) for center in centers])
    labels = np.repeat(np.arange(4), 12)
    contexts = np.zeros((cues.shape[0], 6))
    contexts[:, 0] = 1.0
    model = ComplementaryLearningSystem(cue_dim=14, target_dim=4, context_dim=6, dg_dim=640, rng_seed=13)
    before = float((np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean())
    for cue, context, label in zip(cues, contexts, labels):
        model.learn_episode(cue, int(label), context, replay_steps=0)
    online = float((np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean())
    model.consolidate(steps=400, batch_size=24)
    after = float((np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean())
    assert after >= online, (online, after)
    assert after - before >= 0.50, (before, after)
    return {"cortical_before": before, "cortical_online": online, "cortical_after_replay": after}


def main() -> None:
    results = {
        "dg_decorrelation": validate_dg_decorrelation(),
        "ca3_completion": validate_ca3_completion(),
        "one_shot_recall_and_novelty": validate_one_shot_recall_and_novelty(),
        "cortical_replay_improvement": validate_cortical_replay_improvement(),
    }
    for name, metrics in results.items():
        formatted = " ".join(f"{key}={value:.3f}" for key, value in metrics.items())
        print(f"PASS {name}: {formatted}")


if __name__ == "__main__":
    main()
