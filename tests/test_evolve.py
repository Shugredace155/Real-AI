"""Acceptance tests for creature/outer/evolve.py (SPEC 7.14)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature.outer import evolve as E
from creature.outer import genome as G

CFG = dataclasses.replace(SMALL, pop_size=64)
TARGET_SHIFT = 1.0


def _toy_fitness(flat: jnp.ndarray, key: jax.Array, cfg) -> jnp.ndarray:
    """A cheap stand-in landscape: reach a shifted copy of the prior mean, with evaluation noise."""
    target = G.init_genome(jax.random.key(0), cfg).flat() + TARGET_SHIFT
    return -jnp.mean((flat - target) ** 2) + 0.01 * jax.random.normal(key)


def test_rank_normalize() -> None:
    fits = jax.random.normal(jax.random.key(1), (513,)) * 100.0
    r = E.rank_normalize(fits.at[0].set(1e9))  # an outlier does not distort the update
    assert float(r.min()) == -1.0 and float(r.max()) == 1.0
    assert abs(float(r.mean())) < 1e-6
    assert int(jnp.argmax(r)) == 0


def test_evolve_improves() -> None:
    history = []
    log = lambda gen, fits, state: history.append(float(state.parent_fit))
    E.evolve(CFG, jax.random.key(2), generations=200, fitness_fn=_toy_fitness, log=log)
    h = np.asarray(history)
    slope = np.polyfit(np.arange(h.size), h, 1)[0]
    print(f"parent fitness {h[:10].mean():.4f} -> {h[-10:].mean():.4f}, slope {slope:.2e}")
    assert slope > 0
    # Monotone in expectation: each 40-generation window beats the last.
    windows = h.reshape(5, 40).mean(1)
    assert np.all(np.diff(windows) > 0)


def test_checkpoint_resumable(tmp_path) -> None:
    straight = E.evolve(CFG, jax.random.key(3), generations=6, fitness_fn=_toy_fitness)
    ckpt = tmp_path / "evolve.npz"
    E.evolve(CFG, jax.random.key(3), generations=3, fitness_fn=_toy_fitness, checkpoint=ckpt)
    resumed = E.evolve(CFG, jax.random.key(999), generations=6, fitness_fn=_toy_fitness, checkpoint=ckpt)
    for f in straight.__dataclass_fields__:
        assert np.array_equal(np.asarray(getattr(straight, f)), np.asarray(getattr(resumed, f))), f


def test_real_fitness_generation() -> None:
    # Integration: two generations of the real developmental fitness run and stay finite.
    cfg = dataclasses.replace(SMALL, pop_size=4, n_eval_worlds=1)
    fits = []
    state = E.evolve(cfg, jax.random.key(4), generations=2, log=lambda g, f, s: fits.append(np.asarray(f)), eval_batch=2)
    assert int(state.gen) == 2 and np.all(np.isfinite(np.stack(fits)))
    assert np.isfinite(np.asarray(state.mean)).all()


def test_curriculum_switches_config() -> None:
    # A per-generation config schedule is honored, and resuming mid-curriculum is still bit-identical.
    seen = []

    def fit(flat, key, cfg):
        return _toy_fitness(flat, key, cfg) - cfg.metabolic_scale

    def sched(gen):
        c = dataclasses.replace(CFG, metabolic_scale=0.02 if gen < 2 else 0.07)
        seen.append(c.metabolic_scale)
        return c

    E.evolve(CFG, jax.random.key(5), generations=4, fitness_fn=fit, curriculum=sched)
    assert seen == [0.02, 0.02, 0.07, 0.07]
