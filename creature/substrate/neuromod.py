"""Six neuromodulator channels as leaky integrators, and the knobs they set (SPEC 7.6).

Each channel tracks its own driver signal with an evolved gain and time
constant. DA is tonic plus |valence|: unsigned homeostatic importance, so the
organism keeps learning while things are steady or getting worse (DECISIONS
D31). Neuromodulators are global scalars whose only job is to gate
computation through `knobs`; a channel that sets no knob violates Invariant I5.

The plasticity gate is a product over four channels. Every factor carries an
additive floor, because a multiplicative gate over neuromodulators is
otherwise a chain of vetoes: any one channel at 0 would freeze all learning.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import struct

DA, SHT, NE, ACH, OT, CORT = range(6)
CHANNELS = ('DA', '5HT', 'NE', 'ACh', 'OT', 'CORT')

NM_INIT = 0.5  # every channel's level at birth
NM_MAX = 2.0  # upper clip for every channel
SHT_AROUSAL_REF = 0.5  # arousal above which 5HT is driven down (impulsivity)
CORT_PRED_ERR_THRESHOLD = 0.3  # pred_err above which arousal drives CORT (sustained stress)

DA_FLOOR = 0.10  # additive floor on the DA factor of the plasticity gate
ACH_FLOOR = 0.30  # additive floor on the ACh factor
CORT_FLOOR = 0.30  # lower bound on the CORT suppression factor
CORT_SUPPRESS = 0.35  # plasticity removed per unit CORT, before the floor

# knob mapping: knob = offset + scale · channel
PRECISION_OFFSET = 0.5  # ACh → attention precision weighting
TEMPERATURE_OFFSET, TEMPERATURE_SCALE = 0.1, 0.5  # NE → exploration temperature
DISCOUNT_OFFSET, DISCOUNT_SCALE = 0.5, 0.4  # 5HT → time discount factor, in [0.5, 1.3]
ENCODING_OFFSET = 0.5  # CORT → memory encoding strength


@struct.dataclass
class NeuromodState:
    nm: jnp.ndarray  # (6,) channel levels, always in [0, NM_MAX]


def init_neuromod(g: Any) -> NeuromodState:
    """Neuromodulator state at birth. `g` (the genome) is accepted for interface symmetry."""
    return NeuromodState(nm=jnp.full((6,), NM_INIT))


def neuromod_target(
    valence: jnp.ndarray, arousal: jnp.ndarray, pred_err: jnp.ndarray, social_presence: jnp.ndarray, g: Any
) -> jnp.ndarray:
    """Level each channel relaxes toward, shape (6,).

    Args:
        valence: () signed rate of homeostatic error reduction (affect/core.py).
        arousal: () unsigned arousal in [0, 1].
        pred_err: () total prediction error (substrate/pcn.py).
        social_presence: () caregiver proximity in [0, 1].
        g: genome; uses `nm_gain` (6,) and `da_tonic` (1,).
    Invariant: I7 (every driver comes from body state or the organism's own errors).
    """
    stressed = (pred_err > CORT_PRED_ERR_THRESHOLD).astype(jnp.float32)
    drive = jnp.stack(
        [
            jnp.abs(jnp.clip(valence, -1.0, 1.0)),  # DA: homeostatic importance, unsigned
            jnp.clip(SHT_AROUSAL_REF - arousal, -1.0, 1.0),  # 5HT: patience
            jnp.clip(pred_err, 0.0, 1.0),  # NE: predictive surprise
            jnp.clip(1.0 - pred_err, 0.0, 1.0),  # ACh: expected certainty
            jnp.clip(social_presence, 0.0, 1.0),  # OT: conspecific presence
            jnp.clip(arousal * stressed, 0.0, 1.0),  # CORT: sustained stress
        ]
    )
    target = drive * g.nm_gain
    return target.at[DA].add(g.da_tonic[0])  # tonic floor, unscaled by gain


def step_neuromod(
    state: NeuromodState,
    valence: jnp.ndarray,
    arousal: jnp.ndarray,
    pred_err: jnp.ndarray,
    social_presence: jnp.ndarray,
    g: Any,
    dt: float,
) -> NeuromodState:
    """Integrate the six channels by one timestep. DA is tonic + |valence|.

    Args:
        state: current levels.
        valence, arousal, pred_err, social_presence: see `neuromod_target`.
        g: genome; uses `nm_gain`, `nm_tau` (6,) in seconds, `da_tonic`.
        dt: timestep in seconds.
    Returns:
        Next `NeuromodState`, levels in [0, NM_MAX].
    Invariants: I5, I7, I10.
    """
    target = neuromod_target(valence, arousal, pred_err, social_presence, g)
    # A time constant below dt would make the Euler step overshoot its target.
    tau = jnp.maximum(g.nm_tau, dt)
    nm = state.nm + dt * (target - state.nm) / tau
    return NeuromodState(nm=jnp.clip(nm, 0.0, NM_MAX))


def plasticity_gate(nm: jnp.ndarray) -> jnp.ndarray:
    """Global plasticity gate. A structural floor on every factor keeps it above zero.

    Minimum over nm ∈ [0, 2]^6 is DA_FLOOR · 1 · ACH_FLOOR · CORT_FLOOR = 0.009.
    Invariant: I5 (DA, NE, ACh and CORT all move it).
    """
    da = DA_FLOOR + nm[DA]
    ne = 1.0 + nm[NE]
    ach = ACH_FLOOR + nm[ACH]
    cort = jnp.maximum(1.0 - CORT_SUPPRESS * nm[CORT], CORT_FLOOR)
    return da * ne * ach * cort


def knobs(nm: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Map channels to global computational parameters. Every channel appears.

    Invariant: I5.
    """
    return dict(
        plasticity_rate=plasticity_gate(nm),
        precision=PRECISION_OFFSET + nm[ACH],
        temperature=TEMPERATURE_OFFSET + TEMPERATURE_SCALE * nm[NE],
        discount=DISCOUNT_OFFSET + DISCOUNT_SCALE * nm[SHT],
        social_weight=nm[OT],
        encoding=ENCODING_OFFSET + nm[CORT],
    )
