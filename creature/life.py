"""Scheduling a life: stretches of waking steps punctuated by sleep (SPEC 5.1 step 11).

`organism.step` is one waking millisecond and never sleeps. This module runs
waking stretches with `lax.scan`, and every `sleep_period` steps hands the
organism to memory/sleep.py. Sleep is instantaneous in world time: the world
is paused while the organism consolidates.

Waking statistics are averaged in bins of `bin_steps` so long lives fit in
memory; a bin of 1 is per-step telemetry.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from creature import organism as O
from creature.body import homeostat as hs
from creature.body import sandbox as sb
from creature.config import Config
from creature.memory import sleep as sl

SLEEP_STREAM = 1  # fold_in tag separating sleep randomness from waking randomness


@struct.dataclass
class Twin:
    """A resting body in an identical world: the baseline competence is measured against.

    It shares the organism's world randomness and prediction-error stream, so the
    two differ only in what the organism *does* (DECISIONS D53).
    """

    world: sb.World
    homeo: hs.HomeostatState


def make_twin(world: sb.World, cfg: Config) -> Twin:
    return Twin(world=world, homeo=hs.init_homeostat(cfg))


def _twin_step(twin: Twin, key: jax.Array, pred_err: jnp.ndarray, cfg: Config) -> tuple[Twin, jnp.ndarray]:
    rest = jnp.zeros(cfg.act_dim)
    k_world, _ = jax.random.split(key)  # the same world noise organism.step draws
    world = sb.step_world(twin.world, rest, k_world, cfg)
    homeo = hs.step_homeostat(twin.homeo, rest, world, pred_err, cfg)
    return Twin(world=world, homeo=homeo), O.health(homeo.drives, homeo.alive, cfg)


def awake(
    org: O.Organism, world: sb.World, life_key: jax.Array, n_bins: int, bin_steps: int, cfg: Config, twin: Twin | None = None
) -> tuple[O.Organism, sb.World, dict, Twin | None]:
    """Live n_bins · bin_steps waking steps. Returns per-bin means of the step info.

    With a twin, the stats also hold `advantage` = health − twin health.
    """

    def one_step(c, _):
        o, w, tw = c
        key = O.step_key(life_key, o.t)
        o, w, info = O.step(o, w, key, cfg)
        if tw is not None:
            tw, twin_health = _twin_step(tw, key, info["pred_err"], cfg)
            info = dict(info, advantage=info["health"] - twin_health)
        return (o, w, tw), info

    def one_bin(c, _):
        c, infos = jax.lax.scan(one_step, c, None, length=bin_steps)
        return c, jax.tree.map(lambda x: x.astype(jnp.float32).mean(0), infos)

    (org, world, twin), stats = jax.lax.scan(one_bin, (org, world, twin), None, length=n_bins)
    return org, world, stats, twin


def sleep_now(org: O.Organism, life_key: jax.Array, cfg: Config) -> O.Organism:
    """One night of consolidation (needs a non-empty episodic buffer)."""
    key = jax.random.fold_in(jax.random.fold_in(life_key, SLEEP_STREAM), org.t)
    return org.replace(pcn=sl.sleep(org.pcn, org.episodic, key, cfg), sleep_count=org.sleep_count + 1)


def segments(t0: int, n_steps: int, cfg: Config, extra_breaks: tuple[int, ...] = ()) -> list[tuple[int, int, bool]]:
    """Split [t0, t0 + n_steps) at sleep times and extra breakpoints.

    Returns [(start, length, sleep_after)], where sleep_after means the
    segment ends exactly on a sleep boundary.
    """
    end = t0 + n_steps
    cuts = {t for t in range(t0 + 1, end + 1) if t % cfg.sleep_period == 0}
    cuts |= {b for b in extra_breaks if t0 < b < end}
    cuts.add(end)
    out, start = [], t0
    for c in sorted(cuts):
        out.append((start, c - start, c % cfg.sleep_period == 0))
        start = c
    return out


def live(
    org: O.Organism,
    world: sb.World,
    life_key: jax.Array,
    t0: int,
    n_steps: int,
    bin_steps: int,
    cfg: Config,
    at_break: Callable[[O.Organism, sb.World, int], tuple[O.Organism, sb.World]] | None = None,
    extra_breaks: tuple[int, ...] = (),
    twin: Twin | None = None,
) -> tuple[O.Organism, sb.World, dict]:
    """Live from step t0 for n_steps, sleeping on schedule.

    `t0` must equal `org.t` (passed statically so the schedule is known while
    tracing). `at_break(org, world, t)` is called at each extra breakpoint
    (e.g. a body perturbation; it is applied to the twin's world too).
    Segment lengths must be multiples of bin_steps. Returns the organism,
    world and concatenated per-bin stats.
    """
    stats = []
    for start, length, sleep_after in segments(t0, n_steps, cfg, extra_breaks):
        if length % bin_steps:
            raise ValueError(f"segment of {length} steps is not a multiple of bin_steps={bin_steps}")
        org, world, s, twin = awake(org, world, life_key, length // bin_steps, bin_steps, cfg, twin)
        stats.append(s)
        end = start + length
        if sleep_after:
            org = sleep_now(org, life_key, cfg)
        if at_break is not None and end in extra_breaks:
            org, world = at_break(org, world, end)
            if twin is not None:
                twin = twin.replace(world=at_break(org, twin.world, end)[1])
    return org, world, jax.tree.map(lambda *xs: jnp.concatenate(xs), *stats)
