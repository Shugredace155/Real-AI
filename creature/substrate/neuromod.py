"""Six neuromodulator channels as leaky integrators (SPEC 7.6).

Each channel tracks its own driver signal with an evolved gain and time
constant. Neuromodulators are global scalars; their only job is to gate
computation elsewhere (plasticity here in substrate/plasticity.py; precision,
temperature, discount and social weighting in agent/policy.py). A channel that
gates nothing violates Invariant I5.
"""

from __future__ import annotations

import jax.numpy as jnp

from creature.config import Config

DA, SHT, NE, ACH, OT, CORT = range(6)
CHANNELS = ('DA', '5HT', 'NE', 'ACh', 'OT', 'CORT')


def init_neuromod(cfg: Config) -> jnp.ndarray:
    """Neuromodulator levels at birth, shape (n_neuromod,)."""
    return jnp.asarray(cfg.nm_init, dtype=jnp.float32)


def neuromod_drive(
    valence: jnp.ndarray,
    arousal: jnp.ndarray,
    pred_err: jnp.ndarray,
    social_presence: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """The raw driver of each channel before gain, shape (6,).

    Args:
        valence: () signed rate of homeostatic error reduction (affect/core.py).
        arousal: () unsigned arousal in [0, 1].
        pred_err: () total prediction error (substrate/pcn.py).
        social_presence: () caregiver proximity in [0, 1].
        cfg: global config.
    Invariant: I7 (every driver is derived from body state or the organism's own errors).
    """
    stressed = (pred_err > cfg.cort_pred_err_threshold).astype(arousal.dtype)
    return jnp.stack(
        [
            jnp.clip(valence, -1.0, 1.0),  # DA: reward prediction error
            jnp.clip(cfg.sht_arousal_ref - arousal, -1.0, 1.0),  # 5HT: patience, opposes impulsivity
            jnp.clip(pred_err, 0.0, 1.0),  # NE: surprise / urgency
            jnp.clip(1.0 - pred_err, 0.0, 1.0),  # ACh: expected certainty
            jnp.clip(social_presence, 0.0, 1.0),  # OT: conspecific presence
            jnp.clip(arousal * stressed, 0.0, 1.0),  # CORT: sustained stress
        ]
    )


def step_neuromod(
    nm: jnp.ndarray,
    valence: jnp.ndarray,
    arousal: jnp.ndarray,
    pred_err: jnp.ndarray,
    social_presence: jnp.ndarray,
    nm_gain: jnp.ndarray,
    nm_tau: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """Advance all six channels by one timestep.

    Args:
        nm: (6,) current levels.
        valence, arousal, pred_err, social_presence: see `neuromod_drive`.
        nm_gain: (6,) evolved driver gains (genome).
        nm_tau: (6,) evolved time constants in seconds (genome). Values below
            `cfg.dt` are treated as `cfg.dt`, so an evolved time constant can
            never make the integrator overshoot.
        cfg: global config.
    Returns:
        (6,) levels in [0, cfg.nm_max].
    Invariants: I5 (each channel feeds a knob), I7, I10.
    """
    target = neuromod_drive(valence, arousal, pred_err, social_presence, cfg) * nm_gain
    tau = jnp.maximum(nm_tau, cfg.dt)
    nm = nm + cfg.dt * (target - nm) / tau
    return jnp.clip(nm, 0.0, cfg.nm_max)
