"""Acceptance tests for creature/body/homeostat.py (SPEC 7.2)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from creature.body import homeostat as hs
from creature.body import sandbox as sb
from creature.config import Config

CFG = Config()
REST = jnp.zeros(CFG.act_dim)


def _static_world(key: jax.Array) -> sb.World:
    """A world in which nothing moves on its own: the caregiver is frozen."""
    w = sb.randomize_world(key, CFG)
    zero = jnp.zeros(())
    return w.replace(params=w.params.replace(cg_gain=zero, cg_wander=zero, cg_voc_response=zero))


def _live(
    world: sb.World,
    action: jnp.ndarray,
    steps: int,
    pred_err_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    keep_alive: bool,
) -> tuple[hs.HomeostatState, jnp.ndarray]:
    """Run body + homeostat for `steps`. Returns final state and per-step drives.

    pred_err_fn(t, obs) supplies the prediction error a PCN would report.
    keep_alive refills energy and integrity each step, isolating one drive's
    dynamics from death (a dead body's drives are frozen).
    """

    def body(carry, k):
        w, h = carry
        w = sb.step_world(w, action, k, CFG)
        obs = sb.sense(w, h.drives, CFG)
        h = hs.step_homeostat(h, action, w, pred_err_fn(h.t, obs), CFG)
        if keep_alive:
            h = h.replace(drives=h.drives.at[hs.ENERGY].set(1.0).at[hs.INTEGRITY].set(1.0))
        return (w, h), h.drives

    keys = jax.random.split(jax.random.key(99), steps)
    (_, h), drives = jax.jit(lambda c, ks: jax.lax.scan(body, c, ks))((world, hs.init_homeostat(CFG)), keys)
    return h, drives


def test_drives_bounded() -> None:
    batch, steps = 256, 4096  # ≈ 10^6 random steps
    worlds = jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(jax.random.key(0), batch))
    states = jax.vmap(lambda _: hs.init_homeostat(CFG))(jnp.arange(batch))

    def body(carry, k):
        w, h, lo, hi = carry
        k_a, k_e, k_w = jax.random.split(k, 3)
        a = jax.random.uniform(k_a, (batch, CFG.act_dim), minval=-1.0, maxval=1.0)
        err = 2.0 * jax.random.exponential(k_e, (batch,))  # PCN error is ≥ 0 and unbounded
        w = jax.vmap(lambda w, a, k: sb.step_world(w, a, k, CFG))(w, a, jax.random.split(k_w, batch))
        h = jax.vmap(lambda h, a, w, e: hs.step_homeostat(h, a, w, e, CFG))(h, a, w, err)
        return (w, h, jnp.minimum(lo, h.drives.min()), jnp.maximum(hi, h.drives.max())), None

    init = (worlds, states, jnp.array(jnp.inf), jnp.array(-jnp.inf))
    (_, _, lo, hi), _ = jax.jit(lambda c, ks: jax.lax.scan(body, c, ks))(init, jax.random.split(jax.random.key(1), steps))
    assert float(lo) >= 0.0 and float(hi) <= 1.0


def test_boredom() -> None:
    # Stand-in for the PCN until SPEC 7.4 exists: an exponential running average
    # of the observation predicts the next one, so in a static world its error
    # falls toward 0 — the world "becomes predictable".
    rate = 0.01
    world = _static_world(jax.random.key(2))

    def body(carry, k):
        w, h, pred = carry
        w = sb.step_world(w, REST, k, CFG)
        obs = sb.sense(w, h.drives, CFG)
        err = jnp.mean(jnp.abs(obs - pred))
        h = hs.step_homeostat(h, REST, w, err, CFG)
        h = h.replace(drives=h.drives.at[hs.ENERGY].set(1.0).at[hs.INTEGRITY].set(1.0))
        return (w, h, pred + rate * (obs - pred)), h.drives[hs.NOVELTY]

    init = (world, hs.init_homeostat(CFG), jnp.zeros(sb.obs_dim(CFG)))
    keys = jax.random.split(jax.random.key(3), 100_000)
    _, novelty = jax.jit(lambda c, ks: jax.lax.scan(body, c, ks))(init, keys)
    assert float(novelty[-1]) < 0.1


def test_starvation() -> None:
    w = sb.randomize_world(jax.random.key(4), CFG)
    w = w.replace(food_amount=jnp.zeros(CFG.n_food), params=w.params.replace(food_regrow=jnp.zeros(())))
    # Worst case for survival is resting (lowest cost): 1 / 0.1 per s = 10 s = 10^4 steps.
    h, _ = _live(w, REST, 20_000, lambda t, o: jnp.zeros(()), keep_alive=False)
    assert float(h.drives[hs.ENERGY]) == 0.0
    assert not bool(h.alive)


def test_ingestion_refills_energy() -> None:
    w = sb.randomize_world(jax.random.key(5), CFG)
    w = w.replace(food_pos=w.food_pos.at[0].set(w.agent_pos), hazard_pos=-w.agent_pos)
    eat = jnp.zeros(CFG.act_dim).at[sb.MANIPULATE].set(1.0)
    h0 = hs.init_homeostat(CFG).replace(drives=jnp.asarray(CFG.drive_init).at[hs.ENERGY].set(0.5))
    w1 = sb.step_world(w, eat, jax.random.key(0), CFG)
    h1 = hs.step_homeostat(h0, eat, w1, jnp.zeros(()), CFG)
    assert float(w1.ingest) > 0.0
    assert float(h1.drives[hs.ENERGY]) > 0.5


def test_metabolic_cost_matches_table() -> None:
    def one_hot(i: int) -> jnp.ndarray:
        return jnp.zeros(CFG.act_dim).at[i].set(1.0)

    assert float(hs.metabolic_cost(REST, CFG)) == pytest.approx(CFG.metabolic_cost_rest)
    assert float(hs.metabolic_cost(one_hot(sb.THRUST), CFG)) == pytest.approx(CFG.metabolic_cost_locomote)
    assert float(hs.metabolic_cost(one_hot(sb.MANIPULATE), CFG)) == pytest.approx(CFG.metabolic_cost_manipulate)
    assert float(hs.metabolic_cost(one_hot(sb.VOCALIZE), CFG)) == pytest.approx(CFG.metabolic_cost_vocalize)


def test_every_drive_losable() -> None:
    """SPEC 7.2: every drive can fall to 0 (energy: test_starvation; novelty: test_boredom)."""
    key = jax.random.key(6)
    zero_err = lambda t, o: jnp.zeros(())

    # integrity: rest inside the hazard
    w = sb.randomize_world(key, CFG)
    w = w.replace(hazard_pos=w.agent_pos)
    h, _ = _live(w, REST, 10_000, zero_err, keep_alive=False)
    assert float(h.drives[hs.INTEGRITY]) == 0.0 and not bool(h.alive)

    # temperature: an arena with zero ambient everywhere
    w = sb.randomize_world(key, CFG)
    z = jnp.zeros(())
    w = w.replace(params=w.params.replace(ambient_base=z, ambient_amp=z, cg_warmth=z))
    _, d = _live(w, REST, 50_000, zero_err, keep_alive=True)
    assert float(d[-1, hs.TEMPERATURE]) < 0.01

    # social: caregiver frozen in the far corner
    w = _static_world(key)
    far = CFG.arena_half_width - CFG.spawn_margin
    w = w.replace(agent_pos=jnp.array([-far, -far]), cg_pos=jnp.array([far, far]))
    _, d = _live(w, REST, 20_000, zero_err, keep_alive=True)
    assert float(d[-1, hs.SOCIAL]) < 0.01

    # safety: maximally erratic prediction error (alternating 0 and 1)
    w = sb.randomize_world(key, CFG)
    _, d = _live(w, REST, 30_000, lambda t, o: (t % 2).astype(jnp.float32), keep_alive=True)
    assert float(d[-1, hs.SAFETY]) < 0.01


def test_dead_stays_dead() -> None:
    w = sb.randomize_world(jax.random.key(7), CFG)
    w = w.replace(hazard_pos=w.agent_pos)
    h, d = _live(w, REST, 10_000, lambda t, o: jnp.zeros(()), keep_alive=False)
    assert not bool(h.alive)
    assert bool(jnp.all(d[-1] == d[-100]))


def test_vmappable() -> None:
    batch = 1024
    worlds = jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(jax.random.key(8), batch))
    states = jax.vmap(lambda _: hs.init_homeostat(CFG))(jnp.arange(batch))
    a = jnp.zeros((batch, CFG.act_dim))
    out = jax.vmap(lambda h, a, w, e: hs.step_homeostat(h, a, w, e, CFG))(states, a, worlds, jnp.zeros(batch))
    assert out.drives.shape == (batch, CFG.n_drives)
    assert out.recent_pred_err.shape == (batch, CFG.pred_err_window)
    assert out.alive.shape == (batch,)
