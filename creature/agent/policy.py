"""Active-inference policy over motor primitives (SPEC 7.10).

The organism chooses among a small repertoire of motor primitives
(`Config.motor_primitives`), each held for one policy period. A learned
forward model predicts, for each primitive, how the latent world state, the
homeostatic error vector, caregiver proximity and prediction error will
change over one period. The policy rolls every primitive forward
`policy_horizon` periods and scores its expected free energy:

    G = Σ_h w_h · ( |e_h|₁ + intrinsic_w[0] · pe_h − social_weight · prox_h )

with w_h ∝ discount^h, then samples from softmax(−G / temperature + reflex).
NE sets the temperature, 5HT the discount, OT the social weight (Invariant I5).
Reflex arcs are inherited biases from the interoceptive vector onto actions.

The forward model learns online with a local delta rule (Invariant I2).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from creature.config import Config


@struct.dataclass
class DynParams:
    """Forward-model weights: one linear layer over latent/action features."""

    W: jnp.ndarray  # (n_out, n_feat)
    b: jnp.ndarray  # (n_out,)


@struct.dataclass
class DynPrediction:
    dz: jnp.ndarray  # (z_dim,) change of latent over one period
    de: jnp.ndarray  # (n_drives,) change of homeostatic error vector over one period
    prox: jnp.ndarray  # () mean caregiver proximity over the period
    pe: jnp.ndarray  # () mean prediction error over the period


def n_actions(cfg: Config) -> int:
    return len(cfg.motor_primitives)


def z_dim(cfg: Config) -> int:
    """The latent world state is the first latent PCN layer."""
    return cfg.layer_sizes[0]


def dyn_sizes(cfg: Config) -> tuple[int, int]:
    """(n_out, n_feat) of the forward model."""
    a, z = n_actions(cfg), z_dim(cfg)
    return z + cfg.n_drives + 2, z + a + a * cfg.dyn_interaction_dims


def n_reflex(cfg: Config) -> int:
    """Reflex genes: a bias per action plus a weight from each drive error to each action."""
    return n_actions(cfg) * (1 + cfg.n_drives)


def init_dyn(cfg: Config) -> DynParams:
    """Forward model at birth before the genome overrides it: predicts no change."""
    n_out, n_feat = dyn_sizes(cfg)
    return DynParams(W=jnp.zeros((n_out, n_feat)), b=jnp.zeros(n_out))


def primitive(idx: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Motor vector of primitive `idx`, shape (act_dim,)."""
    return jnp.asarray(cfg.motor_primitives)[idx]


def dyn_features(z: jnp.ndarray, a_idx: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Latent, action one-hot, and their interaction (so outcomes can depend on context × action)."""
    zt = jnp.tanh(z)
    onehot = jax.nn.one_hot(a_idx, n_actions(cfg))
    cross = (onehot[:, None] * zt[None, : cfg.dyn_interaction_dims]).reshape(-1)
    return jnp.concatenate([zt, onehot, cross])


def dyn_predict(dyn: DynParams, z: jnp.ndarray, a_idx: jnp.ndarray, cfg: Config) -> DynPrediction:
    out = dyn.W @ dyn_features(z, a_idx, cfg) + dyn.b
    zd = z_dim(cfg)
    return DynPrediction(
        dz=out[:zd], de=out[zd : zd + cfg.n_drives], prox=out[zd + cfg.n_drives], pe=out[zd + cfg.n_drives + 1]
    )


def dyn_target(dz: jnp.ndarray, de: jnp.ndarray, prox: jnp.ndarray, pe: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([dz, de, prox[None], pe[None]])


def dyn_deltas(dyn: DynParams, z0: jnp.ndarray, a_idx: jnp.ndarray, target: jnp.ndarray, cfg: Config) -> DynParams:
    """Local delta-rule learning signal for one observed period (Invariant I2)."""
    f = dyn_features(z0, a_idx, cfg)
    err = target - (dyn.W @ f + dyn.b)
    return DynParams(W=jnp.outer(err, f), b=err)


def expected_free_energy(
    dyn: DynParams, z: jnp.ndarray, err_vec: jnp.ndarray, knobs: dict, g: Any, cfg: Config
) -> jnp.ndarray:
    """G for every primitive, shape (n_actions,). Lower is better.

    Args:
        dyn: forward model.
        z: (z_dim,) current latent.
        err_vec: (n_drives,) current urgency-weighted homeostatic error.
        knobs: from `neuromod.knobs`; uses discount, social_weight.
        g: endowment; uses intrinsic_w.
        cfg: global config.
    """
    H = cfg.policy_horizon
    weights = knobs["discount"] ** jnp.arange(1, H + 1)
    weights = weights / weights.sum()

    def rollout(a_idx):
        def body(c, _):
            z, e = c
            p = dyn_predict(dyn, z, a_idx, cfg)
            e = e + p.de
            step_g = jnp.abs(e).sum() + g.intrinsic_w[0] * p.pe - knobs["social_weight"] * p.prox
            return (z + p.dz, e), step_g

        _, gs = jax.lax.scan(body, (z, err_vec), None, length=H)
        return jnp.sum(weights * gs)

    return jax.vmap(rollout)(jnp.arange(n_actions(cfg)))


def reflex_logits(err_vec: jnp.ndarray, g: Any, cfg: Config) -> jnp.ndarray:
    """Inherited reflex arcs: bias + drive-error → action weights, shape (n_actions,)."""
    a = n_actions(cfg)
    bias, W = g.reflex[:a], g.reflex[a:].reshape(a, cfg.n_drives)
    return bias + W @ err_vec


def policy_logits(
    dyn: DynParams, z: jnp.ndarray, err_vec: jnp.ndarray, knobs: dict, g: Any, cfg: Config
) -> jnp.ndarray:
    """Action logits: −G / temperature plus reflexes. Invariant I5 (NE, 5HT, OT)."""
    G = expected_free_energy(dyn, z, err_vec, knobs, g, cfg)
    return -G / (knobs["temperature"] + 1e-6) + reflex_logits(err_vec, g, cfg)


def policy(
    dyn: DynParams, z: jnp.ndarray, err_vec: jnp.ndarray, knobs: dict, g: Any, cfg: Config, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a motor primitive.

    Returns:
        (action index, logits).
    Invariants: I5, I9, I10.
    """
    logits = policy_logits(dyn, z, err_vec, knobs, g, cfg)
    return jax.random.categorical(key, logits), logits


def entropy(logits: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(logits)
    return -jnp.sum(p * jnp.log(p + 1e-12))
