"""Acceptance tests for creature/memory/sleep.py (SPEC 7.9)."""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature import life
from creature import organism as O
from creature.body import sandbox as sb
from creature.memory import sleep as sl

CFG = SMALL
PACKAGE = Path(sl.__file__).resolve().parent.parent
GRADIENT_APIS = {"grad", "value_and_grad", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linear_transpose"}
LIFE = jax.random.key(1)


def _lived(steps: int) -> O.Organism:
    """An organism that has been awake `steps` steps (no sleep yet)."""
    org, world = O.spawn(O.default_endowment(jax.random.key(0), CFG), CFG), sb.randomize_world(jax.random.key(2), CFG)
    org, _, _ = jax.jit(lambda o, w: life.awake(o, w, LIFE, steps // 100, 100, CFG))(org, world)
    return org


def _loss_on(org: O.Organism, idx: jnp.ndarray, params=None) -> float:
    p = org.pcn if params is None else params
    return float(sl.replay_loss(p, org.episodic.k[idx], org.episodic.v[idx], CFG))


ORG = _lived(CFG.sleep_period - 100)  # a day's experience, just before bedtime
SLEPT = jax.jit(lambda p, e, k: sl.sleep(p, e, k, CFG))(ORG.pcn, ORG.episodic, jax.random.key(3))


def test_sleep_improves() -> None:
    # On the replayed (salience-weighted) distribution, prediction error falls.
    _, _, idx = ORG.episodic.replay_batch(jax.random.key(4), 512, CFG)
    before, after = _loss_on(ORG, idx), _loss_on(ORG, idx, SLEPT)
    print(f"replayed-distribution surprise {before:.5f} -> {after:.5f}")
    assert after < before


def test_sleep_preserves_recent() -> None:
    # The most recent lived steps (up to 10^4, bounded by capacity) do not get worse by more than 10%.
    n = min(10_000, int(ORG.episodic.fill))
    idx = (int(ORG.episodic.ptr) - 1 - jnp.arange(n)) % ORG.episodic.capacity
    before, after = _loss_on(ORG, idx), _loss_on(ORG, idx, SLEPT)
    print(f"recent-experience surprise {before:.5f} -> {after:.5f}")
    assert after <= 1.10 * before


def test_only_global_pass_here() -> None:
    users = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if names & GRADIENT_APIS:
            users.append(path.relative_to(PACKAGE).as_posix())
    assert users == ["memory/sleep.py"], users


def test_prunes_and_realigns() -> None:
    assert all(float(jnp.min(jnp.where(w == 0, jnp.inf, jnp.abs(w)))) >= CFG.prune_threshold for w in SLEPT.W_down)
    for up, down in zip(SLEPT.W_up, SLEPT.W_down):
        assert np.array_equal(np.asarray(up), np.asarray(down.T))


def test_scheduled_by_life() -> None:
    # live() sleeps exactly at multiples of sleep_period, and the world is paused while asleep.
    org, world = O.spawn(O.default_endowment(jax.random.key(0), CFG), CFG), sb.randomize_world(jax.random.key(2), CFG)
    org, _, stats = life.live(org, world, LIFE, 0, 2 * CFG.sleep_period + 500, 500, CFG)
    assert int(org.sleep_count) == 2
    assert int(org.t) == 2 * CFG.sleep_period + 500
    assert stats["pred_err"].shape == ((2 * CFG.sleep_period + 500) // 500,)
