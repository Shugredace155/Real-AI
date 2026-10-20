"""Tests for creature/organism.py: the awake control loop (SPEC 5.1)."""

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature import organism as O
from creature.body import sandbox as sb

CFG = SMALL
G = O.default_endowment(jax.random.key(0), CFG)
LIFE = jax.random.key(1)


def _run(steps: int, world_key: int = 2):
    org, world = O.spawn(G, CFG), sb.randomize_world(jax.random.key(world_key), CFG)

    def body(c, _):
        o, w = c
        o, w, info = O.step(o, w, O.step_key(LIFE, o.t), CFG)
        return (o, w), info

    return jax.jit(lambda o, w: jax.lax.scan(body, (o, w), None, length=steps))(org, world)


def test_learns_from_first_step() -> None:
    # Invariant I3: no separate training mode; the newborn's first step changes its weights.
    (org, _), _ = _run(1)
    changed = [float(jnp.abs(a - b).max()) > 0 for a, b in zip(jax.tree.leaves(org.pcn), jax.tree.leaves(G.pcn_init))]
    assert any(changed)


def test_deterministic_given_key() -> None:
    a, b = _run(300), _run(300)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_action_held_for_policy_period() -> None:
    _, info = _run(10 * CFG.policy_period)
    a = np.asarray(info["action"]).reshape(-1, CFG.policy_period)
    # The primitive changes only at period boundaries (the last step of each block chooses the next).
    assert all(len(set(row[:-1])) == 1 for row in a[1:])


def test_lives_and_predicts_better() -> None:
    (org, _), info = _run(20_000)
    pe = np.asarray(info["pred_err"])
    assert bool(org.homeo.alive)
    assert pe[-2000:].mean() < pe[:2000].mean()
    assert int(org.episodic.fill) == min(20_000, CFG.episodic_capacity)


def test_vmappable() -> None:
    b = 64  # whole organisms are large; the per-module vmap checks use 1024
    gs = jax.vmap(lambda k: O.default_endowment(k, CFG))(jax.random.split(jax.random.key(3), b))
    orgs = jax.vmap(lambda g: O.spawn(g, CFG))(gs)
    worlds = jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(jax.random.key(4), b))
    keys = jax.random.split(jax.random.key(5), b)
    orgs, worlds, info = jax.vmap(lambda o, w, k: O.step(o, w, k, CFG))(orgs, worlds, keys)
    assert info["pred_err"].shape == (b,)
    assert orgs.t.shape == (b,)
