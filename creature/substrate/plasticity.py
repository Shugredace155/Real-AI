"""Three-factor plasticity with metaplasticity (SPEC 7.5).

Every lifetime weight change while awake goes through `apply_delta`: a local
learning signal (factors 1 and 2: pre- and postsynaptic terms) is scaled by a
global neuromodulatory gate (factor 3) and by a per-synapse learning rate that
shrinks as the synapse is used (metaplasticity). `plastic_update` is the Oja
Hebbian instance of that rule; substrate/pcn.py supplies its own error-driven
deltas to the same machinery.

Metaplasticity is the anti-catastrophic-forgetting mechanism: heavily used
synapses harden, so new learning lands on fresh synapses instead of
overwriting old ones. Passive decay (`cfg.meta_decay`) lets unused synapses
reopen.
"""

from __future__ import annotations

import jax.numpy as jnp

from creature.config import Config
from creature.substrate.neuromod import ACH, CORT, DA, NE


def neuromod_gate(nm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Global plasticity gate from four neuromodulators.

    DA permits learning, NE amplifies it, ACh scales it with expected
    certainty, CORT suppresses it. Zero DA freezes all plasticity.

    Args:
        nm: (6,) neuromodulator levels.
        cfg: global config.
    Returns:
        () gate ≥ 0.
    Invariant: I5 (DA, NE, ACh and CORT are all causal through this gate).
    """
    suppress = jnp.maximum(1.0 - cfg.cort_plasticity_suppression * nm[CORT], 0.0)
    return nm[DA] * (1.0 + nm[NE]) * nm[ACH] * suppress


def effective_lr(log_lr: jnp.ndarray, meta: jnp.ndarray) -> jnp.ndarray:
    """Per-synapse learning rate after metaplastic hardening: exp(log_lr) / (1 + meta)."""
    return jnp.exp(log_lr) / (1.0 + meta)


def apply_delta(
    w: jnp.ndarray,
    log_lr: jnp.ndarray,
    meta: jnp.ndarray,
    delta: jnp.ndarray,
    nm: jnp.ndarray,
    meta_tau: jnp.ndarray,
    cfg: Config,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a local learning signal under the neuromodulatory gate and metaplasticity.

    Args:
        w: weights, any shape.
        log_lr: per-synapse base log learning rate (genome), same shape as `w`.
        meta: per-synapse metaplastic state ≥ 0, same shape as `w`.
        delta: local learning signal, same shape as `w` (e.g. Oja term, PCN error term).
        nm: (6,) neuromodulator levels.
        meta_tau: per-synapse hardening constant (genome), broadcastable to `w`.
        cfg: global config.
    Returns:
        (w, meta) after one timestep.
    Invariants: I2 (purely local), I5, I10.
    """
    gated = neuromod_gate(nm, cfg) * delta
    w = w + cfg.dt * effective_lr(log_lr, meta) * gated
    # Harden by the plasticity the synapse actually expressed, so a gated-off
    # synapse does not harden; it only relaxes back toward fully plastic.
    meta = meta + cfg.dt * (jnp.abs(gated) / meta_tau - meta * cfg.meta_decay)
    return w, meta


def oja_delta(w: jnp.ndarray, pre: jnp.ndarray, post: jnp.ndarray) -> jnp.ndarray:
    """Oja-normalized Hebbian term: post·preᵀ − post²·w (row-wise normalization).

    Args:
        w: (n_post, n_pre) weights.
        pre: (n_pre,) presynaptic activity.
        post: (n_post,) postsynaptic activity.
    Returns:
        (n_post, n_pre) learning signal. The −post²·w term keeps each row's norm bounded.
    """
    return jnp.outer(post, pre) - (post**2)[:, None] * w


def plastic_update(
    w: jnp.ndarray,
    log_lr: jnp.ndarray,
    meta: jnp.ndarray,
    pre: jnp.ndarray,
    post: jnp.ndarray,
    nm: jnp.ndarray,
    meta_tau: jnp.ndarray,
    cfg: Config,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Three-factor Oja Hebbian update with metaplasticity (SPEC 7.5).

    Args:
        w: (n_post, n_pre) weights.
        log_lr, meta: per-synapse, same shape as `w`.
        pre: (n_pre,) presynaptic activity.
        post: (n_post,) postsynaptic activity.
        nm: (6,) neuromodulator levels.
        meta_tau: per-synapse hardening constant, broadcastable to `w`.
        cfg: global config.
    Returns:
        (w, meta).
    Invariants: I2, I5, I10.
    """
    return apply_delta(w, log_lr, meta, oja_delta(w, pre, post), nm, meta_tau, cfg)
