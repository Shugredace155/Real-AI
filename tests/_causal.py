"""Shared Invariant I5 check: every neuromodulator channel moves at least one knob.

A channel is causal if clamping it to 0 vs NM_MAX (others held at 1.0) changes
some entry of `neuromod.knobs` by more than 1e-3. Knobs are consumed by the
PCN (precision, plasticity_rate), the policy (temperature, discount,
social_weight) and episodic memory (encoding); once agent/policy.py exists
this check should also probe the action distribution.
"""

import jax.numpy as jnp

from creature.config import Config
from creature.substrate import neuromod as nmod

CFG = Config()
TOL = 1e-3


def check_channel_causal(channel: int) -> None:
    base = jnp.full(CFG.n_neuromod, 1.0)
    lo = nmod.knobs(base.at[channel].set(0.0))
    hi = nmod.knobs(base.at[channel].set(nmod.NM_MAX))
    moved = [k for k in lo if float(jnp.abs(lo[k] - hi[k])) > TOL]
    assert moved, f"{nmod.CHANNELS[channel]} moves no knob"
