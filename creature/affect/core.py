"""Valence and arousal, computed from the body alone (SPEC 7.7).

Valence is the time derivative of total homeostatic error, never its level
(Invariant I6). Level-based valence saturates: once comfortable, the creature
would want nothing. Derivative-based valence makes pleasure intrinsically
transient and produces anticipation, relief, and the structural
impossibility of satisfaction.

The only inputs are drive vectors and evolved setpoints/urgencies. There is
no label, string or prompt anywhere in the path (Invariant I7).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def homeostatic_error(drives: jnp.ndarray, g: Any) -> jnp.ndarray:
    """Interoceptive vector: urgency-weighted signed distance of each drive from its setpoint.

    Args:
        drives: (n_drives,) drive levels in [0, 1].
        g: genome; uses `setpoint` and `urgency`, each (n_drives,).
    Returns:
        (n_drives,) signed errors.
    Invariant: I7.
    """
    return (drives - g.setpoint) * g.urgency


def error_magnitude(drives: jnp.ndarray, g: Any) -> jnp.ndarray:
    """Total homeostatic error, the scalar valence differentiates. Use at birth to seed
    `prev_err_mag`, so the first valence is a true derivative rather than a spike."""
    return jnp.abs(homeostatic_error(drives, g)).sum()


def step_affect(
    drives: jnp.ndarray, prev_drives: jnp.ndarray, prev_err_mag: jnp.ndarray, g: Any, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One timestep of core affect.

    Args:
        drives: (n_drives,) drives this step.
        prev_drives: (n_drives,) drives last step.
        prev_err_mag: () total homeostatic error last step.
        g: genome; uses `setpoint`, `urgency`.
        dt: timestep in seconds.
    Returns:
        (valence, arousal, err_mag). valence is signed, per second: positive when
        error is falling (pleasure), negative when rising. arousal is in [0, 1].
    Invariants: I6 (valence is −d(err)/dt), I7 (body state only), I10.
    """
    err_mag = error_magnitude(drives, g)
    valence = -(err_mag - prev_err_mag) / dt  # falling error = pleasure
    arousal = jnp.abs(drives - prev_drives).sum() + err_mag
    return valence, jnp.tanh(arousal), err_mag
