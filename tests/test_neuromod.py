"""Acceptance tests for creature/substrate/neuromod.py (SPEC 7.6)."""

import jax
import jax.numpy as jnp
import pytest
from _causal import check_channel_causal

from creature.config import Config
from creature.substrate import neuromod as nmod

CFG = Config()
GAIN = jnp.asarray(CFG.nm_gain_init)
TAU = jnp.asarray(CFG.nm_tau_init)


@pytest.mark.parametrize("channel", range(CFG.n_neuromod), ids=nmod.CHANNELS)
def test_all_six_causal(channel: int) -> None:
    check_channel_causal(channel)


def test_timescales_distinct() -> None:
    assert float(TAU.max() / TAU.min()) >= 10.0
    assert len(set(CFG.nm_tau_init)) == CFG.n_neuromod


def test_bounded() -> None:
    # Adversarial inputs, gains and time constants (including tau below dt).
    k_in, k_g, k_t = jax.random.split(jax.random.key(0), 3)
    gain = 5.0 * jax.random.normal(k_g, (CFG.n_neuromod,))
    tau = jnp.exp(jax.random.uniform(k_t, (CFG.n_neuromod,), minval=-10.0, maxval=3.0))
    inputs = 10.0 * jax.random.normal(k_in, (100_000, 4))

    def body(nm, u):
        nm = nmod.step_neuromod(nm, u[0], u[1], u[2], u[3], gain, tau, CFG)
        return nm, (nm.min(), nm.max())

    _, (lo, hi) = jax.lax.scan(body, nmod.init_neuromod(CFG), inputs)
    assert float(lo.min()) >= 0.0 and float(hi.max()) <= CFG.nm_max
    assert bool(jnp.all(jnp.isfinite(lo))) and bool(jnp.all(jnp.isfinite(hi)))


def test_tracks_driver() -> None:
    # Constant inputs: every channel settles at clip(driver · gain, 0, nm_max).
    v, a, e, s = jnp.array(0.4), jnp.array(0.2), jnp.array(0.5), jnp.array(0.7)

    def body(nm, _):
        return nmod.step_neuromod(nm, v, a, e, s, GAIN, TAU, CFG), None

    nm, _ = jax.lax.scan(body, nmod.init_neuromod(CFG), None, length=200_000)  # 10 × slowest tau
    expected = jnp.clip(nmod.neuromod_drive(v, a, e, s, CFG) * GAIN, 0.0, CFG.nm_max)
    assert jnp.allclose(nm, expected, atol=1e-3)


def test_timescales_order_response() -> None:
    # After a step input, the fastest channel (NE) has moved further toward its
    # target than the slowest (CORT): each channel integrates on its own clock.
    nm = jnp.zeros(CFG.n_neuromod)
    one = jnp.array(1.0)
    for _ in range(100):
        nm = nmod.step_neuromod(nm, one, one, one, one, jnp.ones(CFG.n_neuromod), TAU, CFG)
    assert float(nm[nmod.NE]) > float(nm[nmod.CORT])


def test_vmappable() -> None:
    b = 1024
    x = jnp.linspace(-1.0, 1.0, b)
    nm = jax.vmap(lambda v: nmod.step_neuromod(nmod.init_neuromod(CFG), v, v, v, v, GAIN, TAU, CFG))(x)
    assert nm.shape == (b, CFG.n_neuromod)
