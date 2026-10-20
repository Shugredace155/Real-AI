"""Acceptance tests for creature/substrate/neuromod.py (SPEC 7.6, DECISIONS D31)."""

import jax
import jax.numpy as jnp
import pytest
from _birth import birth_genome
from _causal import check_channel_causal

from creature.config import Config
from creature.substrate import neuromod as nmod

CFG = Config()
G = birth_genome()
ZERO = jnp.array(0.0)


def _hold(valence: float, steps: int, g=G, arousal=0.0, pred_err=0.0, social=0.0):
    """Hold the drivers constant; return the final state and the DA trace."""
    v, a, e, s = (jnp.array(x) for x in (valence, arousal, pred_err, social))

    def body(st, _):
        st = nmod.step_neuromod(st, v, a, e, s, g, CFG.dt)
        return st, st.nm[nmod.DA]

    return jax.lax.scan(body, nmod.init_neuromod(g), None, length=steps)


def test_gate_never_zero() -> None:
    axis = jnp.linspace(0.0, nmod.NM_MAX, 9)
    grid = jnp.stack(jnp.meshgrid(*[axis] * CFG.n_neuromod), -1).reshape(-1, CFG.n_neuromod)  # 9^6 points
    gates = jax.vmap(nmod.plasticity_gate)(grid)
    assert float(gates.min()) > 0.0
    assert float(gates.min()) == pytest.approx(0.009, rel=0.05)


@pytest.mark.parametrize("tonic", [0.2, 0.55, 1.0])
def test_tonic_floor(tonic: float) -> None:
    final, da = _hold(0.0, 100_000, g=birth_genome(da_tonic=tonic))
    assert float(final.nm[nmod.DA]) == pytest.approx(tonic, rel=0.01)
    assert float(da.min()) > nmod.DA_FLOOR


def test_abs_valence() -> None:
    args = (ZERO, jnp.array(0.4), jnp.array(0.7), G)
    up = nmod.neuromod_target(jnp.array(0.5), *args)
    down = nmod.neuromod_target(jnp.array(-0.5), *args)
    assert float(up[nmod.DA]) == float(down[nmod.DA])


def test_ach_at_max_surprise() -> None:
    v, e = ZERO, jnp.array(1.0)

    def body(st, _):
        return nmod.step_neuromod(st, v, ZERO, e, ZERO, G, CFG.dt), None

    st, _ = jax.lax.scan(body, nmod.init_neuromod(G), None, length=100_000)
    assert float(st.nm[nmod.ACH]) < 1e-3  # ACh driven to ~0 by full surprise
    assert float(nmod.plasticity_gate(st.nm)) > 0.0
    assert float(nmod.knobs(st.nm)["plasticity_rate"]) > 0.0


def test_da_never_zero_in_distress() -> None:
    final, _ = _hold(-0.3, 100_000)
    assert float(final.nm[nmod.DA]) > 0.5 * float(G.da_tonic[0])
    assert float(nmod.plasticity_gate(final.nm)) > 0.0


@pytest.mark.parametrize("channel", range(CFG.n_neuromod), ids=nmod.CHANNELS)
def test_all_six_causal(channel: int) -> None:
    check_channel_causal(channel)


def test_timescales_distinct() -> None:
    assert float(G.nm_tau.max() / G.nm_tau.min()) >= 10.0


def test_bounded() -> None:
    # SPEC 7.6: adversarial inputs, gains and time constants (including tau below dt).
    k_in, k_g, k_t, k_d = jax.random.split(jax.random.key(0), 4)
    g = birth_genome().replace(
        nm_gain=5.0 * jax.random.normal(k_g, (CFG.n_neuromod,)),
        nm_tau=jnp.exp(jax.random.uniform(k_t, (CFG.n_neuromod,), minval=-10.0, maxval=3.0)),
        da_tonic=jax.random.uniform(k_d, (1,), minval=-5.0, maxval=5.0),
    )
    inputs = 10.0 * jax.random.normal(k_in, (100_000, 4))

    def body(st, u):
        st = nmod.step_neuromod(st, u[0], u[1], u[2], u[3], g, CFG.dt)
        return st, (st.nm.min(), st.nm.max())

    _, (lo, hi) = jax.lax.scan(body, nmod.init_neuromod(g), inputs)
    assert float(lo.min()) >= 0.0 and float(hi.max()) <= nmod.NM_MAX


def test_vmappable() -> None:
    b = 1024
    x = jnp.linspace(-1.0, 1.0, b)
    st = jax.vmap(lambda v: nmod.step_neuromod(nmod.init_neuromod(G), v, v, v, v, G, CFG.dt))(x)
    assert st.nm.shape == (b, CFG.n_neuromod)
    gates = jax.vmap(nmod.plasticity_gate)(st.nm)
    k = jax.vmap(nmod.knobs)(st.nm)
    assert gates.shape == (b,)
    assert all(v.shape == (b,) for v in k.values())
