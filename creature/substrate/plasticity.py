"""Three-factor plasticity with metaplasticity (SPEC 7.5).

Every lifetime weight change while awake goes through `apply_delta`: a local
learning signal (factors 1 and 2: pre- and postsynaptic terms) is scaled by
the global plasticity gate (factor 3, `neuromod.plasticity_gate`, computed
once per timestep and shared by every synapse) and by a per-synapse learning
rate that shrinks as the synapse is used (metaplasticity). `plastic_update`
is the Oja Hebbian instance of that rule; substrate/pcn.py supplies its own
error-driven deltas to the same machinery.

Metaplasticity is the anti-catastrophic-forgetting mechanism: heavily used
synapses harden, so new learning lands on fresh synapses instead of
overwriting old ones. Hardening is scaled by the gate, so it tracks actual
plasticity events rather than raw co-activation; passive decay lets unused
synapses reopen.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

META_HARDEN = 1.0  # hardening per unit of expressed plasticity, before / meta_tau
META_DECAY = 0.01  # 1/s passive decay of hardening; nonzero so unused synapses reopen
META_CLIP = 1.0e4  # upper bound on metaplastic state
LOG_LR_MIN, LOG_LR_MAX = -12.0, 0.0  # evolved log learning rates are clipped here (lr ≤ 1/s)


def effective_lr(log_lr: jnp.ndarray, meta: jnp.ndarray) -> jnp.ndarray:
    """Per-synapse learning rate after hardening: exp(clip(log_lr)) / (1 + meta)."""
    return jnp.exp(jnp.clip(log_lr, LOG_LR_MIN, LOG_LR_MAX)) / (1.0 + meta)


def apply_delta(
    w: jnp.ndarray,
    log_lr: jnp.ndarray,
    meta: jnp.ndarray,
    delta: jnp.ndarray,
    gate: jnp.ndarray,
    meta_tau: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a local learning signal under the plasticity gate and metaplasticity.

    Args:
        w: weights, any shape.
        log_lr: per-synapse base log learning rate (genome), same shape as `w`.
        meta: per-synapse metaplastic state ≥ 0, same shape as `w`.
        delta: local learning signal, same shape as `w` (Oja term, PCN error term).
        gate: () plasticity gate for this timestep.
        meta_tau: per-synapse hardening constant (genome), broadcastable to `w`.
        dt: timestep in seconds.
    Returns:
        (w, meta) after one timestep.
    Invariants: I2 (purely local), I5, I10.
    """
    w = w + dt * effective_lr(log_lr, meta) * gate * delta
    meta = meta + dt * (gate * jnp.abs(delta) * META_HARDEN / meta_tau - meta * META_DECAY)
    return w, jnp.clip(meta, 0.0, META_CLIP)


def oja_delta(w: jnp.ndarray, pre: jnp.ndarray, post: jnp.ndarray) -> jnp.ndarray:
    """Oja-normalized Hebbian term: post·preᵀ − post²·w, row-wise (DECISIONS D22).

    Args:
        w: (n_post, n_pre) weights.
        pre: (n_pre,) presynaptic activity.
        post: (n_post,) postsynaptic activity.
    Returns:
        (n_post, n_pre) learning signal; the −post²·w term keeps each row's norm bounded.
    """
    return jnp.outer(post, pre) - (post**2)[:, None] * w


def plastic_update(
    w: jnp.ndarray,
    log_lr: jnp.ndarray,
    meta: jnp.ndarray,
    pre: jnp.ndarray,
    post: jnp.ndarray,
    gate: jnp.ndarray,
    g: Any,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Three-factor Oja Hebbian update with metaplasticity (SPEC 7.5).

    Args:
        w: (n_post, n_pre) weights.
        log_lr, meta: per-synapse, same shape as `w`.
        pre: (n_pre,) presynaptic activity.
        post: (n_post,) postsynaptic activity.
        gate: () plasticity gate, computed once per timestep.
        g: genome; uses `meta_tau`, broadcastable to `w`.
        dt: timestep in seconds.
    Returns:
        (w, meta).
    Invariants: I2, I5, I10.
    """
    return apply_delta(w, log_lr, meta, oja_delta(w, pre, post), gate, g.meta_tau, dt)
