"""Homeostatic drives (SPEC 7.2).

Six scalar drives, each in [0, 1]. Each one can fall to 0 (SPEC 7.2: "a drive
that can't fall to 0 is not a drive"). Affect is computed later from these
drives alone (Invariant I7). Setpoints live in the genome and are applied in
affect/core.py, not here, so this module is genome-free.

For `temperature` the value is body temperature rather than satisfaction; how
good a given temperature is depends only on its distance from the evolved
setpoint.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct

from creature.body.sandbox import World, effectors
from creature.config import Config

DRIVES = ('energy', 'temperature', 'integrity', 'social', 'novelty', 'safety')
ENERGY, TEMPERATURE, INTEGRITY, SOCIAL, NOVELTY, SAFETY = range(len(DRIVES))


@struct.dataclass
class HomeostatState:
    """Drive state of the body."""

    drives: jnp.ndarray  # (n_drives,) in [0, 1], ordered as DRIVES
    recent_pred_err: jnp.ndarray  # (pred_err_window,) ring buffer of recent total prediction error
    t: jnp.ndarray  # () int32 step counter; indexes the ring buffer
    alive: jnp.ndarray  # () bool; becomes False forever once energy or integrity reaches 0


def init_homeostat(cfg: Config) -> HomeostatState:
    """Drive state at birth.

    Args:
        cfg: global config (`drive_init`, `pred_err_window`).
    Returns:
        A living `HomeostatState` with an empty prediction-error history.
    """
    return HomeostatState(
        drives=jnp.asarray(cfg.drive_init, dtype=jnp.float32),
        recent_pred_err=jnp.zeros(cfg.pred_err_window),
        t=jnp.zeros((), jnp.int32),
        alive=jnp.ones((), bool),
    )


def metabolic_cost(action: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Energy spent per second for a motor vector.

    The SPEC's METABOLIC_COST table names a cost per discrete action. Actions
    here are continuous, so the cost interpolates linearly from the resting
    cost toward each effector's cost by that effector's intensity; a pure
    single-effector action at full intensity reproduces the table exactly.

    Args:
        action: (act_dim,) motor vector.
        cfg: global config.
    Returns:
        () energy/s.
    """
    _, _, locomote, manipulate, vocalize = effectors(action)
    rest = cfg.metabolic_cost_rest
    return (
        rest
        + (cfg.metabolic_cost_locomote - rest) * locomote
        + (cfg.metabolic_cost_manipulate - rest) * manipulate
        + (cfg.metabolic_cost_vocalize - rest) * vocalize
    )


def step_homeostat(
    state: HomeostatState, action: jnp.ndarray, world: World, pred_err: jnp.ndarray, cfg: Config
) -> HomeostatState:
    """Integrate drive dynamics over one timestep.

    Args:
        state: current drive state.
        action: (act_dim,) motor vector taken this step.
        world: sandbox state *after* this step; supplies ingestion, damage,
            ambient temperature and caregiver proximity.
        pred_err: () total prediction error this step (from the PCN). Feeds
            novelty (information intake) and safety (error variance).
        cfg: global config.
    Returns:
        Next `HomeostatState`. A dead body's drives are frozen.
    Invariants: I7 (drives depend only on body/world physics and the organism's
        own prediction error), I10.
    """
    dt = cfg.dt
    _, _, locomote, _, _ = effectors(action)
    window = state.recent_pred_err.at[state.t % cfg.pred_err_window].set(pred_err)
    energy, temp, _, social, novelty, safety = state.drives

    d_energy = -metabolic_cost(action, cfg) + world.ingest
    d_temp = (world.ambient - temp) * cfg.thermal_k + cfg.thermal_action_gain * locomote
    d_integ = cfg.repair_rate - world.damage
    d_social = (world.proximity - social) * cfg.social_k
    # Boredom: intake falls as the world becomes predictable, and novelty follows it to 0.
    info_intake = jnp.clip(jnp.mean(window) * cfg.info_gain, 0.0, 1.0)
    d_novel = (info_intake - novelty) * cfg.novel_k
    safe_target = jnp.clip(1.0 - cfg.safety_var_gain * jnp.var(window), 0.0, 1.0)
    d_safety = (safe_target - safety) * cfg.safety_k

    rates = jnp.stack([d_energy, d_temp, d_integ, d_social, d_novel, d_safety])
    drives = jnp.clip(state.drives + rates * dt, 0.0, 1.0)
    drives = jnp.where(state.alive, drives, state.drives)
    alive = state.alive & (drives[ENERGY] > 0) & (drives[INTEGRITY] > 0)
    return HomeostatState(drives=drives, recent_pred_err=window, t=state.t + 1, alive=alive)
