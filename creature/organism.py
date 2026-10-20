"""The living organism: endowment, state, and the awake control loop (SPEC 5.1).

`step` is one millisecond of waking life, in the SPEC's order:

    0. the world advances under the action being held
    1. sense            2. predictive coding      3. homeostat
    4. affect           5. neuromodulators        6. knobs
    7. policy           8. self-model             9. plasticity
    10. episodic write

Sleep (11) is scheduled by `live`, between stretches of waking steps, and is
the only place a gradient is ever taken (Invariant I2). There is no training
mode: every step learns (Invariant I3).

The endowment is the phenotype the genome expresses at birth. This module
never imports from creature.outer, so the Phase 2 life cannot reach the
evolutionary machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from creature.affect import core as affect
from creature.agent import policy as pol
from creature.agent import selfmodel as sm
from creature.body import homeostat as hs
from creature.body import sandbox as sb
from creature.config import Config
from creature.memory.episodic import Episodic
from creature.substrate import neuromod as nmod
from creature.substrate import pcn
from creature.substrate import plasticity as pl


@struct.dataclass
class Endowment:
    """Phenotype at birth: everything inherited, in the form the substrate uses."""

    pcn_init: pcn.PCNParams  # initial PCN weights (W_up = W_downᵀ)
    pcn_log_lr: pcn.PCNParams  # per-synapse base log learning rates
    pcn_meta_tau: pcn.PCNParams  # per-synapse hardening constants
    dyn_init: pol.DynParams  # initial forward-model weights
    dyn_log_lr: pol.DynParams
    dyn_meta_tau: pol.DynParams
    nm_gain: jnp.ndarray  # (6,)
    nm_tau: jnp.ndarray  # (6,) seconds
    da_tonic: jnp.ndarray  # (1,)
    setpoint: jnp.ndarray  # (n_drives,)
    urgency: jnp.ndarray  # (n_drives,)
    intrinsic_w: jnp.ndarray  # (n_intrinsic,)
    reflex: jnp.ndarray  # (n_reflex,)


@struct.dataclass
class Organism:
    t: jnp.ndarray  # () int32 steps lived
    g: Endowment
    pcn: pcn.PCNParams
    pcn_meta: pcn.PCNParams
    pcn_state: pcn.PCNState
    dyn: pol.DynParams
    dyn_meta: pol.DynParams
    selfm: sm.SelfParams
    homeo: hs.HomeostatState
    nm: nmod.NeuromodState
    err_mag: jnp.ndarray  # () total homeostatic error last step
    action_idx: jnp.ndarray  # () int32 primitive being held
    action: jnp.ndarray  # (act_dim,) its motor vector
    z0: jnp.ndarray  # (z_dim,) latent at the start of the current policy period
    e0: jnp.ndarray  # (n_drives,) error vector at the start of the period
    a0: jnp.ndarray  # () int32 primitive chosen for the period
    prox_sum: jnp.ndarray  # () caregiver proximity summed over the period
    pe_sum: jnp.ndarray  # () prediction error summed over the period
    self_in: jnp.ndarray  # (self_input_dim,) internal state last step, with efference copy
    pe_fast: jnp.ndarray  # () fast running average of pred_err
    pe_slow: jnp.ndarray  # () slow running average of pred_err
    obs_win: jnp.ndarray  # (seq_len, obs_dim) recent observations
    lat_win: jnp.ndarray  # (seq_len, latent_dim) recent latent states
    episodic: Episodic
    sleep_count: jnp.ndarray  # () int32


def latent_dim(cfg: Config) -> int:
    return sum(cfg.layer_sizes)


def default_endowment(key: jax.Array, cfg: Config) -> Endowment:
    """Birth defaults from Config, before evolution. The genome prior is built from this.

    Invariant I9 (initial weights drawn from `key`).
    """
    params, _ = pcn.init_pcn(key, pcn.pcn_sizes(sb.obs_dim(cfg), cfg), cfg)
    dyn = pol.init_dyn(cfg)
    const = lambda tree, v: jax.tree.map(lambda x: jnp.full_like(x, v), tree)
    return Endowment(
        pcn_init=params,
        pcn_log_lr=const(params, cfg.init_log_lr),
        pcn_meta_tau=const(params, cfg.init_meta_tau),
        dyn_init=dyn,
        dyn_log_lr=const(dyn, cfg.init_log_lr),
        dyn_meta_tau=const(dyn, cfg.init_meta_tau),
        nm_gain=jnp.asarray(cfg.nm_gain_init),
        nm_tau=jnp.asarray(cfg.nm_tau_init),
        da_tonic=jnp.array([sum(cfg.da_tonic_prior) / 2]),
        setpoint=jnp.asarray(cfg.setpoint_init),
        urgency=jnp.asarray(cfg.urgency_init),
        intrinsic_w=jnp.zeros(cfg.n_intrinsic),
        reflex=jnp.zeros(pol.n_reflex(cfg)),
    )


def spawn(g: Endowment, cfg: Config) -> Organism:
    """Birth. Every weight comes from the endowment; everything lived starts empty (Invariant I1)."""
    sizes = pcn.pcn_sizes(sb.obs_dim(cfg), cfg)
    zeros = tuple(jnp.zeros(n) for n in sizes)
    homeo = hs.init_homeostat(cfg)
    nm = nmod.init_neuromod(g)
    zd = pol.z_dim(cfg)
    L = cfg.episodic_seq_len
    i0 = jnp.zeros((), jnp.int32)
    return Organism(
        t=i0,
        g=g,
        pcn=g.pcn_init,
        pcn_meta=jax.tree.map(jnp.zeros_like, g.pcn_init),
        pcn_state=pcn.PCNState(act=zeros, prev_act=zeros),
        dyn=g.dyn_init,
        dyn_meta=jax.tree.map(jnp.zeros_like, g.dyn_init),
        selfm=sm.init_selfmodel(cfg),
        homeo=homeo,
        nm=nm,
        err_mag=affect.error_magnitude(homeo.drives, g),
        action_idx=i0,
        action=pol.primitive(i0, cfg),
        z0=jnp.zeros(zd),
        e0=affect.homeostatic_error(homeo.drives, g),
        a0=i0,
        prox_sum=jnp.zeros(()),
        pe_sum=jnp.zeros(()),
        self_in=sm.self_input(homeo.drives, jnp.zeros(sm.BODY_VEL_DIMS), nm.nm, jnp.zeros(zd), pol.primitive(i0, cfg)),
        pe_fast=jnp.zeros(()),
        pe_slow=jnp.zeros(()),
        obs_win=jnp.zeros((L, sb.obs_dim(cfg))),
        lat_win=jnp.zeros((L, latent_dim(cfg))),
        episodic=Episodic.create(cfg.episodic_capacity, latent_dim(cfg), L * sb.obs_dim(cfg)),
        sleep_count=i0,
    )


def health(drives: jnp.ndarray, alive: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Task score: closeness of the drives to the *species reference* setpoints (Config),
    not the evolved ones, so evolution cannot raise competence by moving its own goalposts."""
    return alive * (1.0 - jnp.mean(jnp.abs(drives - jnp.asarray(cfg.setpoint_init))))


def apply_tree(params, log_lr, meta, deltas, meta_tau, gate, dt):
    """`plasticity.apply_delta` over every synapse of a weight pytree. Returns (params, meta)."""
    leaves, treedef = jax.tree.flatten(params)
    rest = [treedef.flatten_up_to(x) for x in (log_lr, meta, deltas, meta_tau)]
    out = [pl.apply_delta(w, lr, m, d, gate, tau, dt) for w, lr, m, d, tau in zip(leaves, *rest)]
    return treedef.unflatten([o[0] for o in out]), treedef.unflatten([o[1] for o in out])


def mean_effective_lr(log_lr, meta) -> jnp.ndarray:
    """Mean per-synapse learning rate after hardening, over a weight pytree."""
    lrs = jax.tree.leaves(jax.tree.map(pl.effective_lr, log_lr, meta))
    return sum(x.sum() for x in lrs) / sum(x.size for x in lrs)


def _shift(win: jnp.ndarray, new: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([win[1:], new[None]])


def step(org: Organism, world: sb.World, key: jax.Array, cfg: Config) -> tuple[Organism, sb.World, dict]:
    """One millisecond of waking life (SPEC 5.1). Local learning only (Invariant I2).

    Args:
        org: the organism.
        world: its sandbox.
        key: PRNG key for this step; the caller derives it from the life key and t.
        cfg: global config.
    Returns:
        (organism, world, info), where info holds this step's telemetry scalars.
    Invariants: I2, I3, I5, I9, I10.
    """
    g, dt = org.g, cfg.dt
    k_world, k_pol = jax.random.split(key)
    alive = org.homeo.alive

    # 0. the world advances under the action being held
    world = sb.step_world(world, jnp.where(alive, org.action, 0.0), k_world, cfg)
    # 1. sense
    obs = sb.sense(world, org.homeo.drives, cfg)
    # 2. predictive coding, at the precision set by last step's ACh
    precision = nmod.knobs(org.nm.nm)["precision"]
    out, pstate = pcn.pcn_forward(org.pcn, org.pcn_state, obs, precision, cfg)
    # 3. homeostat
    homeo = hs.step_homeostat(org.homeo, org.action, world, out.pred_err, cfg)
    # 4. affect
    valence, arousal, err_mag = affect.step_affect(homeo.drives, org.homeo.drives, org.err_mag, g, dt)
    # 5. neuromodulators
    nm = nmod.step_neuromod(org.nm, valence, arousal, out.pred_err, world.proximity, g, dt)
    # 6. knobs
    kn = nmod.knobs(nm.nm)
    gate = kn["plasticity_rate"] * homeo.alive  # the dead do not learn

    # 7. policy: a new primitive at each period boundary, held in between
    z = pstate.act[1]
    err_vec = affect.homeostatic_error(homeo.drives, g)
    decide = (org.t + 1) % cfg.policy_period == 0
    a_new, logits = pol.policy(org.dyn, z, err_vec, kn, g, cfg, k_pol)
    action_idx = jnp.where(decide, a_new, org.action_idx)
    action = pol.primitive(action_idx, cfg)

    # 8. self-model: did my body do what I expected, and did I do what I expected?
    vel = obs[4 * cfg.n_rays : 4 * cfg.n_rays + sm.BODY_VEL_DIMS]
    next_state = sm.body_state(homeo.drives, vel)
    self_err = sm.self_model(org.selfm, org.self_in, action, next_state)

    # 9. plasticity: every rule is local and shares one gate
    deltas = pcn.pcn_deltas(org.pcn, pstate, out.errors, precision, cfg)
    new_pcn, new_pcn_meta = apply_tree(org.pcn, g.pcn_log_lr, org.pcn_meta, deltas, g.pcn_meta_tau, gate, dt)

    # forward model: learns from each completed period, so it is updated once per period
    prox_sum = org.prox_sum + world.proximity
    pe_sum = org.pe_sum + out.pred_err
    target = pol.dyn_target(z - org.z0, err_vec - org.e0, prox_sum / cfg.policy_period, pe_sum / cfg.policy_period)
    d_dyn = pol.dyn_deltas(org.dyn, org.z0, org.a0, target, cfg)
    new_dyn, new_dyn_meta = apply_tree(
        org.dyn, g.dyn_log_lr, org.dyn_meta, d_dyn, g.dyn_meta_tau, gate * decide, dt * cfg.policy_period
    )

    selfm = sm.self_update(org.selfm, org.self_in, action, next_state, gate, cfg)

    # 10. episodic write: the last seq_len observations, keyed by the latent state at their start
    obs_win = _shift(org.obs_win, obs)
    lat_win = _shift(org.lat_win, jnp.concatenate(pstate.act[1:]))
    full = (org.t + 1 >= cfg.episodic_seq_len).astype(jnp.float32)
    episodic = org.episodic.write(lat_win[0], obs_win.reshape(-1), valence, out.pred_err, kn["encoding"] * full)
    episodic = episodic.tick(dt)

    # learning progress (intrinsic reward): error falling faster than its slow trend
    pe_fast = org.pe_fast + dt / cfg.progress_tau_fast * (out.pred_err - org.pe_fast)
    pe_slow = org.pe_slow + dt / cfg.progress_tau_slow * (out.pred_err - org.pe_slow)

    zero = jnp.zeros(())
    new = org.replace(
        t=org.t + 1,
        pcn=new_pcn,
        pcn_meta=new_pcn_meta,
        pcn_state=pstate,
        dyn=new_dyn,
        dyn_meta=new_dyn_meta,
        selfm=selfm,
        homeo=homeo,
        nm=nm,
        err_mag=err_mag,
        action_idx=action_idx,
        action=action,
        z0=jnp.where(decide, z, org.z0),
        e0=jnp.where(decide, err_vec, org.e0),
        a0=jnp.where(decide, action_idx, org.a0),
        prox_sum=jnp.where(decide, zero, prox_sum),
        pe_sum=jnp.where(decide, zero, pe_sum),
        self_in=sm.self_input(homeo.drives, vel, nm.nm, z, action),
        pe_fast=pe_fast,
        pe_slow=pe_slow,
        obs_win=obs_win,
        lat_win=lat_win,
        episodic=episodic,
    )
    info = dict(
        drives=homeo.drives,
        valence=valence,
        arousal=arousal,
        err_mag=err_mag,
        nm=nm.nm,
        pred_err=out.pred_err,
        self_err=self_err,
        action=action_idx,
        alive=homeo.alive,
        gate=gate,
        plasticity=gate * mean_effective_lr(g.pcn_log_lr, new_pcn_meta),
        health=health(homeo.drives, homeo.alive, cfg),
        intrinsic=jnp.maximum(pe_slow - pe_fast, 0.0),
        entropy=pol.entropy(logits),
    )
    return new, world, info


def step_key(life_key: jax.Array, t: jnp.ndarray) -> jax.Array:
    """The key for step t of a life: a pure function of the life key and the clock, so a
    restored life draws exactly the randomness the uninterrupted one would have."""
    return jax.random.fold_in(life_key, t)
