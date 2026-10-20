"""Shared Invariant I5 probes: for each neuromodulator channel, the knobs it gates.

A probe takes a full neuromodulator vector and returns a scalar computational
quantity. A channel is causal if clamping it to 0 vs `nm_max` changes at least
one of its probes. Channels with no probe yet (their knob lives in a module
not yet built) fail as pending, so I5 cannot read green early.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from creature.config import Config
from creature.substrate import neuromod as nmod
from creature.substrate import plasticity as pl

CFG = Config()


def _plastic_change(nm: jnp.ndarray) -> jnp.ndarray:
    """Total weight change of one synapse matrix over 100 steps under `nm`."""
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    w0 = 0.1 * jax.random.normal(k1, (8, 8))
    pre = jax.random.normal(k2, (100, 8))
    post = jax.random.normal(k3, (100, 8))
    log_lr = jnp.full_like(w0, CFG.init_log_lr)

    def body(c, x):
        w, meta = c
        return pl.plastic_update(w, log_lr, meta, x[0], x[1], nm, CFG.init_meta_tau, CFG), None

    (w, _), _ = jax.lax.scan(body, (w0, jnp.zeros_like(w0)), (pre, post))
    return jnp.sum(jnp.abs(w - w0))


PROBES: dict[int, list[tuple[str, Callable[[jnp.ndarray], jnp.ndarray]]]] = {
    nmod.DA: [("plasticity", _plastic_change)],
    nmod.SHT: [],  # discount, impulse suppression: agent/policy.py (SPEC 7.10)
    nmod.NE: [("plasticity", _plastic_change)],  # also exploration temperature: agent/policy.py
    nmod.ACH: [("plasticity", _plastic_change)],  # also precision: agent/policy.py
    nmod.OT: [],  # caregiver-proximity weight: agent/policy.py (SPEC 7.10)
    nmod.CORT: [("plasticity", _plastic_change)],
}


def check_channel_causal(channel: int) -> None:
    probes = PROBES[channel]
    if not probes:
        pytest.fail(
            f"I5 not yet implemented for {nmod.CHANNELS[channel]}: its knobs live in agent/policy.py (SPEC 7.6, 7.10)",
            pytrace=False,
        )
    base = jnp.full(CFG.n_neuromod, 1.0)
    lo, hi = base.at[channel].set(0.0), base.at[channel].set(CFG.nm_max)
    changed = [name for name, f in probes if not jnp.allclose(f(lo), f(hi))]
    assert changed, f"{nmod.CHANNELS[channel]} changes none of {[n for n, _ in probes]}"
