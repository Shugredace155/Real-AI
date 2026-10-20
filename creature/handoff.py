"""Phase 2: one body, one life (SPEC 7.15).

No evolution. No meta-learning. No resets. The genome is read once, the
organism is born once, and from then on it only lives. A crash is not a
death: the whole life state (organism, world, telemetry trackers, clock) is
checkpointed, and a restarted process restores it and continues from the
same t, drawing exactly the randomness the uninterrupted life would have
(Invariant I4). The only weights ever read from disk are the genome and this
life's own checkpoint (Invariant I1). Nothing in creature.outer except the
genome interface is imported.

    python -m creature.handoff --preset tune --steps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from creature import life
from creature import organism as O
from creature import telemetry as tm
from creature.body import sandbox as sb
from creature.config import Config, preset
from creature.outer.genome import Genome, express

GENOME_FILE = "species_endowment.npz"
LIFE_FILE = "life_state.npz"
WORLD_STREAM = 2  # fold_in tag: the birth world is drawn from its own stream of the birth key


@struct.dataclass
class Life:
    org: O.Organism
    world: sb.World
    twin: life.Twin  # resting baseline for the growth ratio (D53)
    growth: tm.Growth


def birth(genome: Genome, life_key: jax.Array, cfg: Config) -> Life:
    """The one instantiation (Invariant I4)."""
    world = sb.randomize_world(jax.random.fold_in(life_key, WORLD_STREAM), cfg)
    return Life(org=O.spawn(express(genome, cfg), cfg), world=world, twin=life.make_twin(world, cfg), growth=tm.init_growth())


def save_life(lv: Life, path: str | Path) -> None:
    """Checkpoint for crash recovery: every leaf of the life, bit for bit."""
    leaves = jax.tree.leaves(lv)
    np.savez(path, **{f"leaf_{i:05d}": np.asarray(x) for i, x in enumerate(leaves)})


def load_life(path: str | Path, template: Life) -> Life:
    with np.load(path) as data:
        leaves = [jnp.asarray(data[f"leaf_{i:05d}"]) for i in range(len(data.files))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def live_chunk(lv: Life, life_key: jax.Array, n_steps: int, cfg: Config) -> tuple[Life, dict]:
    """n_steps waking steps with per-step telemetry rows. Must not cross a sleep boundary."""

    def body(c, _):
        key = O.step_key(life_key, c.org.t)
        org, world, info = O.step(c.org, c.world, key, cfg)
        twin, twin_health = life._twin_step(c.twin, key, info["pred_err"], cfg)
        growth = tm.update_growth(c.growth, info["health"] - twin_health, c.org.t, cfg)
        r = dict(tm.row(org, info), growth_ratio=growth.ratio, advantage=info["health"] - twin_health)
        return Life(org=org, world=world, twin=twin, growth=growth), r

    return jax.lax.scan(body, lv, None, length=n_steps)


def run(
    genome_path: str | Path, out_dir: str | Path, cfg: Config, max_steps: int | None = None, life_key: jax.Array | None = None
) -> Life:
    """Live (or resume living) until death or until the clock reaches `max_steps`.

    Writes telemetry under out_dir/telemetry and checkpoints to out_dir/life_state.npz
    every `checkpoint_period` steps and on exit.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if life_key is None:
        raise ValueError("run() needs the life key; main() mints it from the birth seed")
    genome = Genome.load(genome_path, cfg)
    lv = birth(genome, life_key, cfg)
    if (out / LIFE_FILE).exists():
        lv = load_life(out / LIFE_FILE, lv)  # crash recovery, not re-instantiation
    store = tm.TelemetryStore(out / "telemetry")
    chunk = jax.jit(lambda lv: live_chunk(lv, life_key, cfg.log_period, cfg))
    sleep = jax.jit(lambda o: life.sleep_now(o, life_key, cfg))

    # PHASE 2. No evolution. No meta-learning. No resets. One body. One life.
    while bool(lv.org.homeo.alive) and (max_steps is None or int(lv.org.t) < max_steps):
        lv, rows = chunk(lv)
        store.write(rows)
        t = int(lv.org.t)
        if t % cfg.sleep_period == 0:
            lv = lv.replace(org=sleep(lv.org))
        if t % cfg.checkpoint_period == 0:
            save_life(lv, out / LIFE_FILE)
    save_life(lv, out / LIFE_FILE)
    return lv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 2: birth (or resume) the one organism and let it live.")
    ap.add_argument("--preset", default="tune", choices=("tune", "full"))
    ap.add_argument("--runs", default="runs/tune", help="directory holding species_endowment.npz")
    ap.add_argument("--life", default="runs/life", help="directory for this life's checkpoint and telemetry")
    ap.add_argument("--steps", type=int, default=None, help="stop when the clock reaches this (default: live until death)")
    args = ap.parse_args(argv)
    cfg = preset(args.preset)
    lv = run(Path(args.runs) / GENOME_FILE, args.life, cfg, args.steps, life_key=jax.random.key(cfg.birth_seed))
    print(f"t = {int(lv.org.t)}, alive = {bool(lv.org.homeo.alive)}, sleeps = {int(lv.org.sleep_count)}, "
          f"growth ratio = {float(lv.growth.ratio):.3f}")


if __name__ == "__main__":
    main()
