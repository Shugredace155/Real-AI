"""Telemetry: what the life looked like, step by step (SPEC 7.16).

`row` turns an organism and its step info into the SPEC's per-timestep
record. `Growth` tracks the Growth Ratio, the project's primary metric, every
`growth_window` steps:

    GROWTH_RATIO = (competence_now − competence_at_birth) / competence_now

with competence = mean health (organism.health) over the first window of
life (birth) and over the most recent complete window (now).

`TelemetryStore` writes chunks of rows to compressed .npz files.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from creature import organism as O
from creature.config import Config
from creature.substrate import pcn

FIELDS = (
    "t", "drives", "valence", "arousal", "err_mag", "nm", "pred_err", "self_err", "action", "alive",
    "weight_norm", "mean_meta", "lr_mean", "episodic_fill", "sleep_count",
)


def row(org: O.Organism, info: dict) -> dict:
    """One timestep's record (SPEC 7.16). Pure; vmappable."""
    metas = jax.tree.leaves(org.pcn_meta)
    return dict(
        t=org.t,
        drives=info["drives"],
        valence=info["valence"],
        arousal=info["arousal"],
        err_mag=info["err_mag"],
        nm=info["nm"],
        pred_err=info["pred_err"],
        self_err=info["self_err"],
        action=info["action"],
        alive=info["alive"],
        weight_norm=pcn.total_weight_norm(org.pcn),
        mean_meta=sum(m.sum() for m in metas) / sum(m.size for m in metas),
        lr_mean=O.mean_effective_lr(org.g.pcn_log_lr, org.pcn_meta),
        episodic_fill=org.episodic.fill,
        sleep_count=org.sleep_count,
    )


@struct.dataclass
class Growth:
    birth_sum: jnp.ndarray  # () health summed over the first window of life
    window_sum: jnp.ndarray  # () health summed over the current window
    birth: jnp.ndarray  # () competence at birth (NaN until the first window completes)
    now: jnp.ndarray  # () competence over the latest complete window (NaN until then)
    ratio: jnp.ndarray  # () latest growth ratio (NaN until two windows exist)


def init_growth() -> Growth:
    nan = jnp.asarray(jnp.nan)
    return Growth(birth_sum=jnp.zeros(()), window_sum=jnp.zeros(()), birth=nan, now=nan, ratio=nan)


def growth_ratio(birth: jnp.ndarray, now: jnp.ndarray) -> jnp.ndarray:
    """Defined for any competence values, in [−1, 1]; −1 when current competence is not positive."""
    ratio = jnp.clip((now - birth) / jnp.maximum(now, 1e-6), -1.0, 1.0)
    return jnp.where(now > 1e-6, ratio, -1.0)  # no positive competence now: nothing was grown


def update_growth(gr: Growth, health: jnp.ndarray, t: jnp.ndarray, cfg: Config) -> Growth:
    """Account for the health of step t (0-based). Pure; vmappable."""
    w = cfg.growth_window
    in_birth = t < w
    window_sum = gr.window_sum + health
    closes = (t + 1) % w == 0
    birth_sum = gr.birth_sum + jnp.where(in_birth, health, 0.0)
    birth = jnp.where(closes & (t + 1 == w), birth_sum / w, gr.birth)
    now = jnp.where(closes, window_sum / w, gr.now)
    ratio = jnp.where(closes & (t + 1 > w), growth_ratio(birth, now), gr.ratio)
    return Growth(birth_sum=birth_sum, window_sum=jnp.where(closes, 0.0, window_sum), birth=birth, now=now, ratio=ratio)


class TelemetryStore:
    """Append-only directory of compressed chunks, one file per flushed stretch of life."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: dict) -> Path:
        t0 = int(np.asarray(rows["t"]).reshape(-1)[0])
        path = self.dir / f"telemetry_{t0:012d}.npz"
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in rows.items()})
        return path

    def read(self) -> dict:
        files = sorted(self.dir.glob("telemetry_*.npz"))
        if not files:
            return {}
        chunks = [dict(np.load(f)) for f in files]
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
