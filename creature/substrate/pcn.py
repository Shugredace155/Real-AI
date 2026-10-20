"""Hierarchical predictive coding (SPEC 7.4).

Layer 0 is the sensory layer, clamped to the observation. Layers 1..L hold
latent value nodes; `Config.layer_sizes` gives their widths, bottom to top.
Each layer l < L is predicted from the layer above and from its own previous
timestep:

    pred_l = W_down[l] · φ(act_{l+1}) + W_lat[l] · φ(prev_act_l) + b[l]
    err_l  = act_l − pred_l

Per timestep: a top-down sweep sets each latent layer to its prior
prediction, then a few relaxation iterations move the latent layers to
explain the observation using only adjacent-layer errors. Bottom-up errors
travel through W_up, a separate matrix that receives the transpose of
W_down's update (Kolen–Pollack), so no weight is ever read across layers
and no gradient is ever taken. Weight updates are local outer products of a
layer's error with the activity that predicted it (Whittington & Bogacz 2017).

Invariant I2: nothing in this module computes a gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from creature.config import Config


@struct.dataclass
class PCNParams:
    """Synaptic weights. Index l refers to the predicted layer (0..L-1)."""

    W_down: tuple[jnp.ndarray, ...]  # (n_l, n_{l+1}) top-down generative weights
    W_up: tuple[jnp.ndarray, ...]  # (n_{l+1}, n_l) bottom-up error weights, tracks W_down[l]ᵀ
    W_lat: tuple[jnp.ndarray, ...]  # (n_l, n_l) lateral weights from the previous timestep
    b: tuple[jnp.ndarray, ...]  # (n_l,) biases


@struct.dataclass
class PCNState:
    """Value nodes. act[0] is the last observation; act[L] is the top latent."""

    act: tuple[jnp.ndarray, ...]  # (n_l,) for l = 0..L, current timestep
    prev_act: tuple[jnp.ndarray, ...]  # (n_l,) for l = 0..L, previous timestep (temporal context)


@struct.dataclass
class PCNOut:
    """Result of one forward pass."""

    prediction: jnp.ndarray  # (n_0,) prior prediction of the observation, made before seeing it
    errors: tuple[jnp.ndarray, ...]  # (n_l,) for l = 0..L-1, post-relaxation errors (the learning signal)
    pred_err: jnp.ndarray  # () surprise: mean squared prior sensory error; feeds affect and drives


def _phi(x: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    return jnp.tanh(x) if cfg.pcn_nonlinearity == "tanh" else x


def _dphi(x: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    return 1.0 - jnp.tanh(x) ** 2 if cfg.pcn_nonlinearity == "tanh" else jnp.ones_like(x)


def pcn_sizes(obs_dim: int, cfg: Config) -> tuple[int, ...]:
    """Layer widths bottom to top: the observation layer, then `cfg.layer_sizes`."""
    return (obs_dim,) + tuple(cfg.layer_sizes)


def init_pcn(key: jax.Array, sizes: tuple[int, ...], cfg: Config) -> tuple[PCNParams, PCNState]:
    """Birth weights and zeroed value nodes.

    W_down is Gaussian with std `pcn_init_scale / sqrt(fan_in)`; W_up starts as
    its exact transpose; lateral weights and biases start at 0.

    Args:
        key: PRNG key.
        sizes: layer widths, bottom (sensory) to top.
        cfg: global config.
    Returns:
        (params, state).
    Invariant: I9.
    """
    keys = jax.random.split(key, len(sizes) - 1)
    W_down = tuple(
        cfg.pcn_init_scale / jnp.sqrt(sizes[l + 1]) * jax.random.normal(keys[l], (sizes[l], sizes[l + 1]))
        for l in range(len(sizes) - 1)
    )
    params = PCNParams(
        W_down=W_down,
        W_up=tuple(w.T for w in W_down),
        W_lat=tuple(jnp.zeros((n, n)) for n in sizes[:-1]),
        b=tuple(jnp.zeros(n) for n in sizes[:-1]),
    )
    zeros = tuple(jnp.zeros(n) for n in sizes)
    return params, PCNState(act=zeros, prev_act=zeros)


def _predict(params: PCNParams, act: tuple, prev_act: tuple, l: int, cfg: Config) -> jnp.ndarray:
    return (
        params.W_down[l] @ _phi(act[l + 1], cfg)
        + params.W_lat[l] @ _phi(prev_act[l], cfg)
        + params.b[l]
    )


def pcn_forward(
    params: PCNParams,
    state: PCNState,
    x: jnp.ndarray,
    precision: jnp.ndarray,
    cfg: Config,
    clamp_top: jnp.ndarray | None = None,
    n_steps: int | None = None,
) -> tuple[PCNOut, PCNState]:
    """One timestep: top-down sweep, then local relaxation against the observation.

    Args:
        params: weights.
        state: value nodes from the previous timestep.
        x: (n_0,) observation; clamps the sensory layer.
        precision: () or (n_layers - 1,) precision weighting of errors (gated by
            ACh via the policy knobs). Scales both inference and learning.
        cfg: global config.
        clamp_top: optional (n_L,) value to clamp the top layer to. The organism
            leaves it free; supervised tests clamp it to an input.
        n_steps: relaxation iterations; defaults to `cfg.pcn_inference_steps`.
    Returns:
        (PCNOut, new state). The new state's prev_act is this step's context.
    Invariants: I2 (no gradients; adjacent-layer errors only), I10.
    """
    n_layers = len(params.W_down)
    steps = cfg.pcn_inference_steps if n_steps is None else n_steps
    prec = jnp.broadcast_to(precision, (n_layers,))
    prev = state.act

    # 1. top-down sweep: each latent layer starts at its prior prediction
    top = prev[n_layers] if clamp_top is None else clamp_top
    act = [None] * (n_layers + 1)
    act[n_layers] = top
    for l in reversed(range(1, n_layers)):
        act[l] = _predict(params, act, prev, l, cfg)
    act[0] = x
    prior_pred = _predict(params, act, prev, 0, cfg)

    # 2. relaxation: latent layers descend the prediction-error energy using
    #    their own error and the error they cause one layer below (via W_up).
    def relax(act: list, _) -> tuple[list, None]:
        err = [act[l] - _predict(params, act, prev, l, cfg) for l in range(n_layers)]
        new = list(act)
        for l in range(1, n_layers + 1):
            own = -prec[l] * err[l] if l < n_layers else -cfg.pcn_top_prior * act[l]
            below = _dphi(act[l], cfg) * (params.W_up[l - 1] @ (prec[l - 1] * err[l - 1]))
            new[l] = act[l] + cfg.pcn_inference_rate * (own + below)
        if clamp_top is not None:
            new[n_layers] = clamp_top
        return new, None

    act, _ = jax.lax.scan(relax, act, None, length=steps)
    errors = tuple(act[l] - _predict(params, act, prev, l, cfg) for l in range(n_layers))
    out = PCNOut(prediction=prior_pred, errors=errors, pred_err=jnp.mean((x - prior_pred) ** 2))
    return out, PCNState(act=tuple(act), prev_act=prev)


def pcn_deltas(params: PCNParams, state: PCNState, errors: tuple, precision: jnp.ndarray, cfg: Config) -> PCNParams:
    """Local learning signals for every weight, shaped like `params`.

    Each predicted layer's precision-weighted error is multiplied by the
    activity that produced the prediction. W_up receives the transpose of
    W_down's signal; both decay at `pcn_weight_decay`, so W_up converges to
    W_down[l]ᵀ even after sleep changes W_down alone.

    Args:
        params: weights.
        state: value nodes returned by `pcn_forward` for this timestep.
        errors: post-relaxation errors from `pcn_forward`.
        precision: as passed to `pcn_forward`.
        cfg: global config.
    Returns:
        A `PCNParams` of per-second deltas.
    Invariant: I2 (outer products of locally available quantities).
    """
    n_layers = len(params.W_down)
    prec = jnp.broadcast_to(precision, (n_layers,))
    eps = [prec[l] * errors[l] for l in range(n_layers)]
    above = [_phi(state.act[l + 1], cfg) for l in range(n_layers)]
    decay = cfg.pcn_weight_decay
    return PCNParams(
        W_down=tuple(jnp.outer(eps[l], above[l]) - decay * params.W_down[l] for l in range(n_layers)),
        W_up=tuple(jnp.outer(above[l], eps[l]) - decay * params.W_up[l] for l in range(n_layers)),
        W_lat=tuple(jnp.outer(eps[l], _phi(state.prev_act[l], cfg)) for l in range(n_layers)),
        b=tuple(eps),
    )


def pcn_update(
    params: PCNParams, state: PCNState, errors: tuple, lr: jnp.ndarray, precision: jnp.ndarray, cfg: Config
) -> PCNParams:
    """Plain local update: params + dt · lr · deltas.

    The organism's awake loop instead routes `pcn_deltas` through
    substrate/plasticity.apply_delta, which adds the neuromodulatory gate and
    metaplasticity; this entry point is the ungated rule used to check the
    learning algorithm itself.

    Args:
        params, state, errors, precision: see `pcn_deltas`.
        lr: () learning rate per second.
        cfg: global config.
    Returns:
        Updated params.
    Invariant: I2.
    """
    d = pcn_deltas(params, state, errors, precision, cfg)
    return jax.tree.map(lambda p, g: p + cfg.dt * lr * g, params, d)


def total_weight_norm(params: PCNParams) -> jnp.ndarray:
    """L2 norm over every weight; logged by telemetry as `weight_norm`."""
    return jnp.sqrt(sum(jnp.sum(w**2) for w in jax.tree.leaves(params)))
