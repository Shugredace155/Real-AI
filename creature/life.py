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

    It shares the organism's world randomness, so the two differ only in what the
    organism *does* (DECISIONS D53). Having no brain, it is given a fixed reference
    surprise (`twin_pred_err`), so an organism that bores itself into a dark room
    scores worse than the twin on novelty (D58).
    """

    world: sb.World
    homeo: hs.HomeostatState


def make_twin(world: sb.World, cfg: Config) -> Twin:
    return Twin(world=world, homeo=hs.init_homeostat(cfg))


def twin_step(twin: Twin, key: jax.Array, cfg: Config) -> tuple[Twin, jnp.ndarray]:
    """Advance the twin one step with the organism's step key. Returns (twin, twin health)."""
    rest = jnp.zeros(cfg.act_dim)
    k_world, _ = jax.random.split(key)  # the same world noise organism.step draws
    world = sb.step_world(twin.world, rest, k_world, cfg)
    homeo = hs.step_homeostat(twin.homeo, rest, world, jnp.asarray(cfg.twin_pred_err), cfg)
    return Twin(world=world, homeo=homeo), O.health(homeo.drives, homeo.alive, cfg)


def make_probe(world: sb.World, key: jax.Array, cfg: Config) -> jnp.ndarray:
    """A fixed set of short clips of this world, recorded at birth by a scripted body.

    The body drives forward while sweeping its heading; every `probe_stride` steps
    a clip of `probe_clip_len` consecutive frames (with birth drives) is kept.
    Returns (probe_clips, probe_clip_len, obs_dim). Learning is measured on these
    clips, so an organism that never moves cannot score by making its own corner
    predictable (D58).
    """
    drives = jnp.asarray(cfg.drive_init)
    n = cfg.probe_clips * cfg.probe_stride

    def body(w, x):
        t, k = x
        action = jnp.zeros(cfg.act_dim).at[0].set(1.0).at[1].set(jnp.sin(2 * jnp.pi * t / (4 * cfg.probe_stride)))
        w = sb.step_world(w, action, k, cfg)
        return w, sb.sense(w, drives, cfg)

    _, obs = jax.lax.scan(body, world, (jnp.arange(n), jax.random.split(key, n)))
    return obs.reshape(cfg.probe_clips, cfg.probe_stride, -1)[:, : cfg.probe_clip_len]


def probe_error(params, probe: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Mean surprise of the PCN predicting each probe clip frame by frame, as when awake,
    from a blank latent state; the first `probe_warmup` frames only build context."""
    from creature.substrate import pcn

    sizes = tuple(w.shape[0] for w in params.W_down) + (params.W_down[-1].shape[1],)
    blank = tuple(jnp.zeros(n) for n in sizes)

    def clip(frames):
        def body(st, x):
            out, st = pcn.pcn_forward(params, st, x, jnp.array(1.0), cfg)
            return st, out.pred_err

        _, errs = jax.lax.scan(body, pcn.PCNState(act=blank, prev_act=blank), frames)
        return errs[cfg.probe_warmup :].mean()

    return jax.vmap(clip)(probe).mean()


def awake(
    org: O.Organism,
    world: sb.World,
    life_key: jax.Array,
    n_bins: int,
    bin_steps: int,
    cfg: Config,
    twin: Twin | None = None,
    probe: jnp.ndarray | None = None,
) -> tuple[O.Organism, sb.World, dict, Twin | None]:
    """Live n_bins · bin_steps waking steps. Returns per-bin means of the step info.

    With a twin, the stats also hold `advantage` = health − twin health. With a
    probe, they hold `probe_err`, the PCN's error on the probe views at the end of each bin.
    """

    def one_step(c, _):
        o, w, tw = c
        key = O.step_key(life_key, o.t)
        o, w, info = O.step(o, w, key, cfg)
        if tw is not None:
            tw, twin_health = twin_step(tw, key, cfg)
            info = dict(info, advantage=info["health"] - twin_health)
        return (o, w, tw), info

    def one_bin(c, _):
        c, infos = jax.lax.scan(one_step, c, None, length=bin_steps)
        stats = jax.tree.map(lambda x: x.astype(jnp.float32).mean(0), infos)
        if probe is not None:
            stats = dict(stats, probe_err=probe_error(c[0].pcn, probe, cfg))
        return c, stats

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
    probe: jnp.ndarray | None = None,
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
        org, world, s, twin = awake(org, world, life_key, length // bin_steps, bin_steps, cfg, twin, probe)
        stats.append(s)
        end = start + length
        if sleep_after:
            org = sleep_now(org, life_key, cfg)
        if at_break is not None and end in extra_breaks:
            org, world = at_break(org, world, end)
            if twin is not None:
                twin = twin.replace(world=at_break(org, twin.world, end)[1])
    return org, world, jax.tree.map(lambda *xs: jnp.concatenate(xs), *stats)
