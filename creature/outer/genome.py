"""The genome: the entire interface between evolution and life (SPEC 7.12).

A flat vector of genes, small enough to evolve. Fields are stored in an
unconstrained encoding so one mutation scale suits them all; `express` decodes
them into the `Endowment` the organism is born with:

    init_w      raw initial weights of every inherited synapse (PCN W_down,
                W_lat, biases; forward model). W_up is not a gene: it is born
                as W_downᵀ (DECISIONS D19).
    log_lr      per-synapse log learning rate (clipped in plasticity.py)
    meta_tau    per-synapse log hardening constant      -> exp (clipped)
    nm_gain     neuromodulator driver gains             -> as is
    nm_tau      log neuromodulator time constants       -> exp (clipped)
    setpoint    drive setpoints                         -> clip [0, 1]
    urgency     drive urgencies                         -> max(·, 0)
    intrinsic_w intrinsic weights                       -> as is
    reflex      reflex arcs                             -> as is
    da_tonic    tonic dopamine                          -> clip [0, NM_MAX]

Lived, never inherited: fast weights, metaplastic state, episodic memory,
the self-model, everything the organism comes to know.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.flatten_util import ravel_pytree

from creature import organism as O
from creature.agent import policy as pol
from creature.body import sandbox as sb
from creature.config import Config
from creature.substrate import neuromod as nmod
from creature.substrate import pcn


@struct.dataclass
class Inherited:
    """The synapses whose birth weights, rates and hardening are genes."""

    W_down: tuple[jnp.ndarray, ...]
    W_lat: tuple[jnp.ndarray, ...]
    b: tuple[jnp.ndarray, ...]
    dyn: pol.DynParams


def _template(cfg: Config) -> Inherited:
    """Zero-valued Inherited tree; only its shapes matter (no randomness needed)."""
    sizes = pcn.pcn_sizes(sb.obs_dim(cfg), cfg)
    return Inherited(
        W_down=tuple(jnp.zeros((sizes[l], sizes[l + 1])) for l in range(len(sizes) - 1)),
        W_lat=tuple(jnp.zeros((n, n)) for n in sizes[:-1]),
        b=tuple(jnp.zeros(n) for n in sizes[:-1]),
        dyn=pol.init_dyn(cfg),
    )


def _unravel(cfg: Config):
    return ravel_pytree(_template(cfg))[1]


class Genome(struct.PyTreeNode):
    init_w: jnp.ndarray  # (P,)
    log_lr: jnp.ndarray  # (P,)
    meta_tau: jnp.ndarray  # (P,) log-encoded
    nm_gain: jnp.ndarray  # (6,)
    nm_tau: jnp.ndarray  # (6,) log-encoded
    setpoint: jnp.ndarray  # (D,)
    urgency: jnp.ndarray  # (D,)
    intrinsic_w: jnp.ndarray  # (K,)
    reflex: jnp.ndarray  # (R,)
    da_tonic: jnp.ndarray  # (1,)

    def flat(self) -> jnp.ndarray:
        return jnp.concatenate([getattr(self, f) for f in FIELDS])

    @classmethod
    def from_flat(cls, v: jnp.ndarray, cfg: Config | None = None) -> Genome:
        cfg = Config() if cfg is None else cfg
        out, start = {}, 0
        for f, n in zip(FIELDS, field_sizes(cfg)):
            out[f] = v[start : start + n]
            start += n
        return cls(**out)

    def save(self, path: str | Path) -> None:
        np.savez(path, flat=np.asarray(self.flat()))

    @classmethod
    def load(cls, path: str | Path, cfg: Config | None = None) -> Genome:
        """The only weights a Phase 2 life ever reads from disk (Invariant I1)."""
        with np.load(path) as data:
            return cls.from_flat(jnp.asarray(data["flat"]), cfg)


FIELDS = tuple(Genome.__dataclass_fields__)


def n_synapses(cfg: Config) -> int:
    return int(ravel_pytree(_template(cfg))[0].size)


def field_sizes(cfg: Config) -> tuple[int, ...]:
    P = n_synapses(cfg)
    D = cfg.n_drives
    return (P, P, P, cfg.n_neuromod, cfg.n_neuromod, D, D, cfg.n_intrinsic, pol.n_reflex(cfg), 1)


def genome_dim(cfg: Config) -> int:
    return sum(field_sizes(cfg))


def init_genome(key: jax.Array, cfg: Config) -> Genome:
    """The genome prior: birth defaults from Config, random initial weights, da_tonic ~ U(prior).

    Invariant I9.
    """
    k_w, k_da = jax.random.split(key)
    e = O.default_endowment(k_w, cfg)
    w, _ = ravel_pytree(Inherited(W_down=e.pcn_init.W_down, W_lat=e.pcn_init.W_lat, b=e.pcn_init.b, dyn=e.dyn_init))
    P = w.size
    lo, hi = cfg.da_tonic_prior
    return Genome(
        init_w=w,
        log_lr=jnp.full(P, cfg.init_log_lr),
        meta_tau=jnp.full(P, jnp.log(cfg.init_meta_tau)),
        nm_gain=jnp.asarray(cfg.nm_gain_init),
        nm_tau=jnp.log(jnp.asarray(cfg.nm_tau_init)),
        setpoint=jnp.asarray(cfg.setpoint_init),
        urgency=jnp.asarray(cfg.urgency_init),
        intrinsic_w=jnp.zeros(cfg.n_intrinsic),
        reflex=jnp.zeros(pol.n_reflex(cfg)),
        da_tonic=jax.random.uniform(k_da, (1,), minval=lo, maxval=hi),
    )


def _pcn_tree(inh: Inherited) -> pcn.PCNParams:
    return pcn.PCNParams(W_down=inh.W_down, W_up=tuple(w.T for w in inh.W_down), W_lat=inh.W_lat, b=inh.b)


LOG_GENE_BOUND = 20.0  # log-encoded genes are clipped here before exp, so no time constant is 0 or inf


def _exp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(jnp.clip(x, -LOG_GENE_BOUND, LOG_GENE_BOUND))


def express(genome: Genome, cfg: Config) -> O.Endowment:
    """Decode the genome into the phenotype at birth (glossary: Endowment). Pure and deterministic."""
    unravel = _unravel(cfg)
    w, lr, tau = unravel(genome.init_w), unravel(genome.log_lr), unravel(_exp(genome.meta_tau))
    return O.Endowment(
        pcn_init=_pcn_tree(w),
        pcn_log_lr=_pcn_tree(lr),
        pcn_meta_tau=_pcn_tree(tau),
        dyn_init=w.dyn,
        dyn_log_lr=lr.dyn,
        dyn_meta_tau=tau.dyn,
        nm_gain=genome.nm_gain,
        nm_tau=_exp(genome.nm_tau),
        da_tonic=jnp.clip(genome.da_tonic, 0.0, nmod.NM_MAX),
        setpoint=jnp.clip(genome.setpoint, 0.0, 1.0),
        urgency=jnp.maximum(genome.urgency, 0.0),
        intrinsic_w=genome.intrinsic_w,
        reflex=genome.reflex,
    )
