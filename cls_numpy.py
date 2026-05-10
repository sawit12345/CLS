"""Biologically grounded, cheap NumPy Complementary Learning System.

The model keeps the major CLS commitments explicit while staying small enough
for a single CPU NumPy file: entorhinal cue/context input, dentate-gyrus sparse
pattern separation, CA3 recurrent Hebbian pattern completion, CA1-style
heteroassociation/comparator, and slow neocortical semantic consolidation by
interleaved replay. It is a rate-coded systems model, not a spiking simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


def unit_rows(x: Array, eps: float = 1e-12) -> Array:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def softmax(x: Array) -> Array:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z = x - x.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def top_k_binary(x: Array, k: int) -> Array:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    k = max(1, min(k, x.shape[1]))
    idx = np.argpartition(x, -k, axis=1)[:, -k:]
    out = np.zeros_like(x)
    out[np.arange(x.shape[0])[:, None], idx] = 1.0 / np.sqrt(k)
    return out


def one_hot(labels: Array, classes: int) -> Array:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    y = np.zeros((labels.size, classes), dtype=float)
    y[np.arange(labels.size), labels] = 1.0
    return y


@dataclass
class DentateGyrus:
    """Sparse pattern separator with global inhibition."""

    ec_dim: int
    dg_dim: int
    sparsity: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.k = max(1, int(round(self.dg_dim * self.sparsity)))
        self.perforant_path = self.rng.normal(0.0, 1.0 / np.sqrt(self.ec_dim), (self.ec_dim, self.dg_dim))
        self.running_activity = np.zeros(self.dg_dim)

    def encode(self, ec_state: Array, novelty: float = 1.0) -> Array:
        drive = unit_rows(ec_state) @ self.perforant_path
        inhibition = (1.0 - novelty) * self.running_activity.reshape(1, -1)
        code = top_k_binary(drive - inhibition, self.k)
        self.running_activity = 0.995 * self.running_activity + 0.005 * code.mean(axis=0)
        return code


@dataclass
class CA3AutoassociativeMemory:
    """Recurrent CA3 memory using normalized Hebbian outer products."""

    dim: int
    sparsity: float
    decay: float = 0.999

    def __post_init__(self) -> None:
        self.k = max(1, int(round(self.dim * self.sparsity)))
        self.recurrent = np.zeros((self.dim, self.dim), dtype=float)
        self.trace_count = 0

    def store(self, dg_code: Array, strength: float) -> None:
        for code in np.atleast_2d(dg_code):
            self.recurrent *= self.decay
            self.recurrent += strength * np.outer(code, code)
            np.fill_diagonal(self.recurrent, 0.0)
            self.trace_count += 1

    def complete(self, partial_code: Array, steps: int = 3) -> Array:
        state = np.atleast_2d(np.asarray(partial_code, dtype=float))
        for _ in range(steps):
            state = top_k_binary(state @ self.recurrent + 0.35 * state, self.k)
        return state


@dataclass
class CA1EntorhinalBinder:
    """CA1/EC heteroassociation and mismatch estimation."""

    ca3_dim: int
    cue_dim: int
    context_dim: int
    target_dim: int
    decay: float = 0.9995

    def __post_init__(self) -> None:
        self.to_cue = np.zeros((self.cue_dim, self.ca3_dim), dtype=float)
        self.to_context = np.zeros((self.context_dim, self.ca3_dim), dtype=float)
        self.to_target = np.zeros((self.target_dim, self.ca3_dim), dtype=float)

    def bind(self, ca3_code: Array, cue: Array, context: Array, target: Array, strength: float) -> None:
        ca3_code = np.atleast_2d(ca3_code)
        cue = unit_rows(cue)
        context = unit_rows(context)
        target = np.atleast_2d(target)
        for c3, cu, ctx, tgt in zip(ca3_code, cue, context, target):
            self.to_cue = self.decay * self.to_cue + strength * np.outer(cu, c3)
            self.to_context = self.decay * self.to_context + strength * np.outer(ctx, c3)
            self.to_target = self.decay * self.to_target + strength * np.outer(tgt, c3)

    def reconstruct(self, ca3_code: Array) -> tuple[Array, Array, Array]:
        ca3_code = np.atleast_2d(ca3_code)
        cue = unit_rows(ca3_code @ self.to_cue.T)
        context = unit_rows(ca3_code @ self.to_context.T)
        target = softmax(ca3_code @ self.to_target.T)
        return cue, context, target

    def mismatch(self, ca3_code: Array, cue: Array, context: Array, target: Array | None = None) -> Array:
        pred_cue, pred_context, pred_target = self.reconstruct(ca3_code)
        cue_error = 1.0 - np.sum(pred_cue * unit_rows(cue), axis=1)
        context_error = 1.0 - np.sum(pred_context * unit_rows(context), axis=1)
        if target is None:
            return np.clip(0.5 * (cue_error + context_error), 0.0, 1.0)
        target = np.atleast_2d(target)
        target_error = 1.0 - np.sum(pred_target * target, axis=1)
        return np.clip((cue_error + context_error + target_error) / 3.0, 0.0, 1.0)


@dataclass
class SlowNeocortex:
    """Low-cost semantic learner trained gradually by online input and replay."""

    input_dim: int
    hidden_dim: int
    target_dim: int
    learning_rate: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.w1 = self.rng.normal(0.0, 1.0 / np.sqrt(self.input_dim), (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = self.rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), (self.hidden_dim, self.target_dim))
        self.b2 = np.zeros(self.target_dim)

    def predict(self, x: Array) -> Array:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        h = np.tanh(x @ self.w1 + self.b1)
        return softmax(h @ self.w2 + self.b2)

    def train(self, x: Array, target: Array, learning_rate_scale: float = 1.0) -> float:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        target = np.atleast_2d(np.asarray(target, dtype=float))
        h = np.tanh(x @ self.w1 + self.b1)
        pred = softmax(h @ self.w2 + self.b2)
        n = x.shape[0]
        delta2 = (pred - target) / n
        delta1 = (delta2 @ self.w2.T) * (1.0 - h * h)
        lr = self.learning_rate * learning_rate_scale
        self.w2 -= lr * (h.T @ delta2 + 1e-4 * self.w2)
        self.b2 -= lr * delta2.sum(axis=0)
        self.w1 -= lr * (x.T @ delta1 + 1e-4 * self.w1)
        self.b1 -= lr * delta1.sum(axis=0)
        return float(-(target * np.log(pred + 1e-12)).sum(axis=1).mean())


class ComplementaryLearningSystem:
    """Hippocampal fast learning plus neocortical slow learning."""

    def __init__(
        self,
        cue_dim: int,
        target_dim: int,
        *,
        context_dim: int = 8,
        dg_dim: int = 384,
        dg_sparsity: float = 0.04,
        cortical_hidden_dim: int = 96,
        cortical_learning_rate: float = 0.025,
        rng_seed: int = 0,
    ) -> None:
        self.cue_dim = cue_dim
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.rng = np.random.default_rng(rng_seed)
        self.dg = DentateGyrus(cue_dim + context_dim, dg_dim, dg_sparsity, self._rng())
        self.ca3 = CA3AutoassociativeMemory(dg_dim, dg_sparsity)
        self.ca1 = CA1EntorhinalBinder(dg_dim, cue_dim, context_dim, target_dim)
        self.cortex = SlowNeocortex(cue_dim + context_dim, cortical_hidden_dim, target_dim, cortical_learning_rate, self._rng())
        self.episode_cues = np.empty((0, cue_dim), dtype=float)
        self.episode_contexts = np.empty((0, context_dim), dtype=float)
        self.episode_targets = np.empty((0, target_dim), dtype=float)
        self.episode_ca3 = np.empty((0, dg_dim), dtype=float)
        self.episode_novelty = np.empty(0, dtype=float)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(int(self.rng.integers(0, 2**32 - 1)))

    def default_context(self, n: int) -> Array:
        context = np.zeros((n, self.context_dim), dtype=float)
        context[:, 0] = 1.0
        return context

    def ec_state(self, cue: Array, context: Array | None = None) -> Array:
        cue = unit_rows(cue)
        if context is None:
            context = self.default_context(cue.shape[0])
        context = unit_rows(context)
        return np.hstack([cue, context])

    def _target_matrix(self, label_or_target: int | Iterable[int] | Array, n: int) -> Array:
        if np.isscalar(label_or_target):
            return one_hot(np.full(n, int(label_or_target)), self.target_dim)
        arr = np.asarray(label_or_target)
        if arr.ndim == 1 and arr.size == n and np.issubdtype(arr.dtype, np.integer):
            return one_hot(arr, self.target_dim)
        return np.atleast_2d(arr.astype(float))

    def encode_episode(self, cue: Array, context: Array | None = None) -> tuple[Array, Array]:
        ec = self.ec_state(cue, context)
        first_pass = self.dg.encode(ec, novelty=1.0)
        completed = self.ca3.complete(first_pass)
        return ec, completed

    def novelty(self, cue: Array, context: Array | None = None, target: Array | None = None) -> Array:
        cue = unit_rows(cue)
        context = self.default_context(cue.shape[0]) if context is None else unit_rows(context)
        _, completed = self.encode_episode(cue, context)
        if self.ca3.trace_count == 0:
            return np.ones(cue.shape[0])
        return self.ca1.mismatch(completed, cue, context, target)

    def learn_episode(
        self,
        cue: Array,
        label_or_target: int | Iterable[int] | Array,
        context: Array | None = None,
        replay_steps: int = 6,
    ) -> Array:
        cue = unit_rows(cue)
        context = self.default_context(cue.shape[0]) if context is None else unit_rows(context)
        target = self._target_matrix(label_or_target, cue.shape[0])
        novelty = self.novelty(cue, context, target)
        ec = self.ec_state(cue, context)
        dg_code = self.dg.encode(ec, novelty=float(novelty.mean()))
        ca3_code = self.ca3.complete(dg_code)

        hippocampal_strength = 0.25 + 0.85 * novelty
        cortical_scale = 0.15 + 0.85 * (1.0 - novelty)
        for c3, cu, ctx, tgt, h_lr, c_lr in zip(ca3_code, cue, context, target, hippocampal_strength, cortical_scale):
            self.ca3.store(c3.reshape(1, -1), strength=float(h_lr))
            self.ca1.bind(c3.reshape(1, -1), cu.reshape(1, -1), ctx.reshape(1, -1), tgt.reshape(1, -1), strength=float(h_lr))
            self.cortex.train(self.ec_state(cu.reshape(1, -1), ctx.reshape(1, -1)), tgt.reshape(1, -1), float(c_lr))

        self.episode_cues = np.vstack([self.episode_cues, cue])
        self.episode_contexts = np.vstack([self.episode_contexts, context])
        self.episode_targets = np.vstack([self.episode_targets, target])
        self.episode_ca3 = np.vstack([self.episode_ca3, ca3_code])
        self.episode_novelty = np.concatenate([self.episode_novelty, novelty])
        self.consolidate(steps=replay_steps, batch_size=min(24, max(1, self.episode_cues.shape[0])))
        return novelty

    def hippocampal_recall(self, cue: Array, context: Array | None = None) -> Array:
        _, ca3_code = self.encode_episode(cue, context)
        if self.episode_ca3.shape[0] > 0:
            sims = ca3_code @ self.episode_ca3.T
            weights = softmax(sims / 0.08)
            return weights @ self.episode_targets
        _, _, target = self.ca1.reconstruct(ca3_code)
        return target

    def cortical_predict(self, cue: Array, context: Array | None = None) -> Array:
        return self.cortex.predict(self.ec_state(cue, context))

    def predict(self, cue: Array, context: Array | None = None) -> Array:
        novelty = self.novelty(cue, context).reshape(-1, 1)
        hippocampal = self.hippocampal_recall(cue, context)
        cortical = self.cortical_predict(cue, context)
        hippocampal_gate = np.clip(0.25 + 0.65 * novelty, 0.25, 0.9)
        mixed = hippocampal_gate * hippocampal + (1.0 - hippocampal_gate) * cortical
        return mixed / np.maximum(mixed.sum(axis=1, keepdims=True), 1e-12)

    def classify(self, cue: Array, context: Array | None = None) -> Array:
        return np.argmax(self.predict(cue, context), axis=1)

    def replay_batch(self, batch_size: int) -> tuple[Array, Array, Array]:
        if self.episode_cues.shape[0] == 0:
            return self.episode_cues, self.episode_contexts, self.episode_targets
        priority = self.episode_novelty + 0.05
        priority = priority / priority.sum()
        idx = self.rng.choice(self.episode_cues.shape[0], size=batch_size, replace=True, p=priority)
        return self.episode_cues[idx], self.episode_contexts[idx], self.episode_targets[idx]

    def consolidate(self, steps: int = 64, batch_size: int = 16) -> float:
        loss = 0.0
        if self.episode_cues.shape[0] == 0:
            return loss
        for _ in range(steps):
            cue, context, target = self.replay_batch(batch_size)
            loss = self.cortex.train(self.ec_state(cue, context), target, learning_rate_scale=1.0)
        return loss


def demo() -> None:
    rng = np.random.default_rng(11)
    centers = unit_rows(
        np.array(
            [
                [-1.0, -1.0, 0.2, 0.0, 0.1, -0.2],
                [1.0, -1.0, -0.1, 0.1, -0.2, 0.0],
                [0.0, 1.0, -1.0, 0.7, 0.1, 0.2],
            ]
        )
    )
    cues = np.vstack([center + 0.10 * rng.normal(size=(10, 6)) for center in centers])
    cues = unit_rows(cues)
    labels = np.repeat(np.arange(3), 10)
    contexts = np.zeros((cues.shape[0], 8))
    contexts[:, 0] = 1.0

    model = ComplementaryLearningSystem(cue_dim=6, target_dim=3, context_dim=8, rng_seed=5)
    cortical_before = (np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean()
    for cue, context, label in zip(cues, contexts, labels):
        model.learn_episode(cue, int(label), context, replay_steps=3)

    one_shot_accuracy = (model.classify(cues, contexts) == labels).mean()
    cortical_after_online = (np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean()
    model.consolidate(steps=240, batch_size=18)
    cortical_after_replay = (np.argmax(model.cortical_predict(cues, contexts), axis=1) == labels).mean()

    degraded = unit_rows(cues[:6] + 0.12 * rng.normal(size=(6, 6)))
    completion_accuracy = (model.classify(degraded, contexts[:6]) == labels[:6]).mean()
    probe_a = unit_rows(np.array([[1.0, 0.4, 0.1, 0.0, -0.1, 0.2]]))
    probe_b = unit_rows(np.array([[0.7, 0.6, -0.2, 0.1, 0.0, 0.1]]))
    dg_raw_overlap = float((probe_a @ probe_b.T)[0, 0])
    dg_a = model.dg.encode(model.ec_state(probe_a, contexts[:1]), novelty=1.0)
    dg_b = model.dg.encode(model.ec_state(probe_b, contexts[:1]), novelty=1.0)
    dg_overlap = float((dg_a @ dg_b.T)[0, 0])

    print(f"episodes={model.episode_cues.shape[0]} ca3_traces={model.ca3.trace_count}")
    print(f"cortical_accuracy_before={cortical_before:.3f}")
    print(f"hippocampal_one_shot_accuracy={one_shot_accuracy:.3f}")
    print(f"cortical_accuracy_after_online={cortical_after_online:.3f}")
    print(f"cortical_accuracy_after_replay={cortical_after_replay:.3f}")
    print(f"degraded_cue_completion_accuracy={completion_accuracy:.3f}")
    print(f"raw_overlap={dg_raw_overlap:.3f} dg_sparse_overlap={dg_overlap:.3f}")


if __name__ == "__main__":
    demo()
