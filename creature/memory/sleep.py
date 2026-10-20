"""Sleep: offline consolidation from the episodic buffer (SPEC 7.9).

The only place in the package where a gradient is taken (Invariant I2).
Waking learning is local; sleeping learning replays salient episodes and
trains the predictive-coding network end to end, backpropagating through
each replayed observation sequence (through time) and through its inference
relaxation. Afterwards near-zero synapses are pruned, and the feedback
weights W_up are re-aligned to W_downᵀ (DECISIONS D19), which sleep, being
global, is allowed to do directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from creature.config import Config
from creature.memory.episodic import Episodic
from creature.substrate import pcn


def _unpack_key(key: jnp.ndarray, cfg: Config) -> tuple[jnp.ndarray, ...]:
    out, start = [], 0
    for n in cfg.layer_sizes:
        out.append(key[start : start + n])
        start += n
    return tuple(out)


def replay_loss(params: pcn.PCNParams, keys: jnp.ndarray, values: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Mean surprise of the PCN re-living replayed sequences.

    Each trace starts from its stored latent state and its first observation,
    then predicts each following observation in turn, exactly as when awake.

    Args:
        params: PCN weights.
        keys: (B, latent_dim) latent state at the start of each trace.
        values: (B, seq_len · obs_dim) observation sequences.
        cfg: global config.
    Returns:
        () mean prior prediction error over every replayed step.
    """
    seq_len = cfg.episodic_seq_len
    obs_dim = values.shape[-1] // seq_len

    def one(k, v):
        seq = v.reshape(seq_len, obs_dim)
        act = (seq[0],) + _unpack_key(k, cfg)
        state = pcn.PCNState(act=act, prev_act=act)

        def body(st, x):
            out, st = pcn.pcn_forward(params, st, x, jnp.array(1.0), cfg)
            return st, out.pred_err

        _, errs = jax.lax.scan(body, state, seq[1:])
        return errs.mean()

    return jax.vmap(one)(keys, values).mean()


def consolidate_bptt(params: pcn.PCNParams, keys: jnp.ndarray, values: jnp.ndarray, cfg: Config) -> pcn.PCNParams:
    """One clipped gradient step on the replay loss, through time and through inference."""
    grads = jax.grad(replay_loss)(params, keys, values, cfg)
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, cfg.sleep_grad_clip / (norm + 1e-12))
    return jax.tree.map(lambda p, g: p - cfg.sleep_lr * scale * g, params, grads)


def prune(params: pcn.PCNParams, cfg: Config) -> pcn.PCNParams:
    """Remove near-zero generative and lateral synapses; re-align feedback weights to W_downᵀ."""
    cut = lambda w: jnp.where(jnp.abs(w) < cfg.prune_threshold, 0.0, w)
    W_down = tuple(cut(w) for w in params.W_down)
    return params.replace(W_down=W_down, W_lat=tuple(cut(w) for w in params.W_lat), W_up=tuple(w.T for w in W_down))


def sleep(params: pcn.PCNParams, episodic: Episodic, rng: jax.Array, cfg: Config) -> pcn.PCNParams:
    """Offline consolidation. Local rules while awake; global passes while asleep.

    Args:
        params: PCN weights at bedtime.
        episodic: the buffer to replay from (must hold at least one trace).
        rng: PRNG key for replay sampling, split once per consolidation step.
        cfg: global config.
    Returns:
        Consolidated, pruned PCN weights.
    Invariants: I2 (the one sanctioned global pass), I9.
    """

    def body(p, k):
        keys, values, _ = episodic.replay_batch(k, cfg.sleep_replay_batch, cfg)
        return prune(consolidate_bptt(p, keys, values, cfg), cfg), None

    params, _ = jax.lax.scan(body, params, jax.random.split(rng, cfg.sleep_consolidation_steps))
    return params
