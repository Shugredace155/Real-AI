"""Acceptance tests for creature/body/sandbox.py (SPEC 7.3)."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from creature.body import sandbox as sb
from creature.config import Config

CFG = Config()
BATCH = 4096


def _batched_worlds(key: jax.Array, n: int) -> sb.World:
    return jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(key, n))


def _rollout(key: jax.Array, steps: int) -> tuple[sb.World, jnp.ndarray]:
    """One world, random actions; everything derived from `key`."""
    k_world, k_run = jax.random.split(key)
    world = sb.randomize_world(k_world, CFG)

    def body(w, k):
        k_act, k_step = jax.random.split(k)
        a = jax.random.uniform(k_act, (CFG.act_dim,), minval=-1.0, maxval=1.0)
        w = sb.step_world(w, a, k_step, CFG)
        return w, sb.sense(w, jnp.zeros(CFG.n_drives), CFG)

    return jax.jit(lambda w, ks: jax.lax.scan(body, w, ks))(world, jax.random.split(k_run, steps))


def test_throughput() -> None:
    steps = 100
    worlds = _batched_worlds(jax.random.key(0), BATCH)
    drives = jnp.zeros((BATCH, CFG.n_drives))

    def body(w, k):
        k_act, k_step = jax.random.split(k)
        a = jax.random.uniform(k_act, (BATCH, CFG.act_dim), minval=-1.0, maxval=1.0)
        w = jax.vmap(lambda w, a, k: sb.step_world(w, a, k, CFG))(w, a, jax.random.split(k_step, BATCH))
        obs = jax.vmap(lambda w, d: sb.sense(w, d, CFG))(w, drives)
        return w, obs[:, 0]

    run = jax.jit(lambda w, k: jax.lax.scan(body, w, jax.random.split(k, steps)))
    jax.block_until_ready(run(worlds, jax.random.key(1)))  # compile
    start = time.perf_counter()
    jax.block_until_ready(run(worlds, jax.random.key(2)))
    rate = BATCH * steps / (time.perf_counter() - start)
    print(f"sandbox throughput: {rate:,.0f} env-steps/sec at batch {BATCH}")
    assert rate >= 20_000


def test_determinism() -> None:
    a = _rollout(jax.random.key(7), 500)
    b = _rollout(jax.random.key(7), 500)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    c = _rollout(jax.random.key(8), 500)
    assert not np.array_equal(np.asarray(a[1]), np.asarray(c[1]))


def test_worlds_differ() -> None:
    w = _batched_worlds(jax.random.key(3), 1000)
    layout = jnp.concatenate(
        [
            w.agent_pos,
            w.obj_pos.reshape(1000, -1),
            w.food_pos.reshape(1000, -1),
            w.cg_pos,
            w.hazard_pos,
        ],
        axis=1,
    )
    distinct = np.unique(np.round(np.asarray(layout), 3), axis=0).shape[0]
    assert distinct >= 990


def test_vmappable() -> None:
    w = _batched_worlds(jax.random.key(4), BATCH)
    assert w.agent_pos.shape == (BATCH, 2)
    assert w.obj_pos.shape == (BATCH, CFG.n_objects, 2)
    a = jnp.zeros((BATCH, CFG.act_dim))
    w2 = jax.vmap(lambda w, a, k: sb.step_world(w, a, k, CFG))(w, a, jax.random.split(jax.random.key(5), BATCH))
    assert jax.tree.map(jnp.shape, w2) == jax.tree.map(jnp.shape, w)
    obs = jax.vmap(lambda w: sb.sense(w, jnp.zeros(CFG.n_drives), CFG))(w2)
    assert obs.shape == (BATCH, sb.obs_dim(CFG))


def test_observation_is_low_level() -> None:
    # Rays carry only depth and colour in [0, 1]; nothing else from the world is exposed.
    _, obs = _rollout(jax.random.key(6), 200)
    rays = obs[:, : 4 * CFG.n_rays]
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert bool(jnp.all((rays >= 0.0) & (rays <= 1.0)))


def test_bodies_stay_in_arena() -> None:
    w, _ = _rollout(jax.random.key(9), 3000)
    lim = CFG.arena_half_width
    assert bool(jnp.all(jnp.abs(w.agent_pos) <= lim))
    assert bool(jnp.all(jnp.abs(w.obj_pos) <= lim))
    assert bool(jnp.all(jnp.abs(w.cg_pos) <= lim))
