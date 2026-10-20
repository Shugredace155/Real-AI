"""Self-model: predicts the organism's own next action and body state (SPEC 7.11).

The input is the organism's own internal state from the previous step,
including its current body state and the action it just issued (an efference copy). From it the
self-model predicts the next action and the next body state (drives and
body-frame velocity). Mismatch is self-surprise, tracked separately from
world surprise (pred_err). Because the efference copy is an input, changes
the organism caused itself become predictable while external perturbations
do not: this is how it tells self-caused from world-caused change.

Lived, never inherited: born at zero and learned online by a local delta rule
(Invariant I2).
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct

from creature.config import Config


@struct.dataclass
class SelfParams:
    Wa: jnp.ndarray  # (act_dim, n_in) next-action readout
    ba: jnp.ndarray  # (act_dim,)
    Ws: jnp.ndarray  # (n_state, n_in) next-body-state readout
    bs: jnp.ndarray  # (n_state,)


BODY_VEL_DIMS = 2  # body-frame velocity components in the predicted state


def self_input_dim(cfg: Config) -> int:
    """drives + body velocity + neuromodulators + latent + efference copy of the last action."""
    return cfg.n_drives + BODY_VEL_DIMS + cfg.n_neuromod + cfg.layer_sizes[0] + cfg.act_dim


def self_state_dim(cfg: Config) -> int:
    return cfg.n_drives + BODY_VEL_DIMS


def init_selfmodel(cfg: Config) -> SelfParams:
    """Born knowing nothing about itself."""
    n_in, n_s = self_input_dim(cfg), self_state_dim(cfg)
    return SelfParams(
        Wa=jnp.zeros((cfg.act_dim, n_in)), ba=jnp.zeros(cfg.act_dim), Ws=jnp.zeros((n_s, n_in)), bs=jnp.zeros(n_s)
    )


def self_input(
    drives: jnp.ndarray, body_vel: jnp.ndarray, nm: jnp.ndarray, z: jnp.ndarray, last_action: jnp.ndarray
) -> jnp.ndarray:
    """The current body state is part of the input: the next state is predicted from it plus the efference copy."""
    return jnp.concatenate([drives, body_vel, nm, jnp.tanh(z), last_action])


def body_state(drives: jnp.ndarray, body_vel: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([drives, body_vel])


def self_net(params: SelfParams, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(predicted next action, predicted next body state) from the previous internal state."""
    return params.Wa @ state + params.ba, params.Ws @ state + params.bs


def self_model(params: SelfParams, state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray) -> jnp.ndarray:
    """Self-surprise: mean |action error| + mean |state error|. SPEC 7.11.

    Args:
        params: self-model weights.
        state: (self_input_dim,) internal state at t−1, including the efference copy.
        action: (act_dim,) action actually issued at t.
        next_state: (self_state_dim,) body state actually reached at t.
    Returns:
        () self-surprise ≥ 0.
    """
    pred_action, pred_state = self_net(params, state)
    action_err = jnp.abs(pred_action - action).mean()
    state_err = jnp.abs(pred_state - next_state).mean()
    return action_err + state_err


def self_update(
    params: SelfParams, state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray, gate: jnp.ndarray, cfg: Config
) -> SelfParams:
    """Local delta rule, scaled by the shared plasticity gate (Invariant I2)."""
    pred_action, pred_state = self_net(params, state)
    ea, es = action - pred_action, next_state - pred_state
    step = cfg.dt * cfg.selfmodel_lr * gate
    return SelfParams(
        Wa=params.Wa + step * jnp.outer(ea, state),
        ba=params.ba + step * ea,
        Ws=params.Ws + step * jnp.outer(es, state),
        bs=params.bs + step * es,
    )
