# Real-AI — Project Creature

A synthetic organism that learns continuously from its own experience, with
affect that changes computation. Not an LLM wrapper: no transformer, no
pretrained weights, no training corpus. See [SPEC.md](SPEC.md) for the
specification and [DECISIONS.md](DECISIONS.md) for every ambiguity resolved
and every deviation, with the reason.

## Layout

```
creature/
  config.py          all hyperparameters; presets "full" (SPEC) and "tune" (CPU)
  body/              sandbox.py (2D world), homeostat.py (six drives)
  substrate/         pcn.py (predictive coding), plasticity.py, neuromod.py
  affect/core.py     valence = −d(error)/dt, arousal
  agent/             policy.py (active inference), selfmodel.py
  memory/            episodic.py (salience-tagged buffer), sleep.py (the only backprop)
  organism.py        the waking control loop (SPEC 5.1)
  life.py            waking stretches + sleep scheduling, resting-twin baseline
  outer/             genome.py, fitness.py, evolve.py (Phase 1)
  telemetry.py       per-step records and the Growth Ratio
  handoff.py         Phase 2: one birth, one continuous life
tests/               one file per module + test_invariants.py (the ten invariants)
```

## Running

```bash
pip install -e ".[test]"
pytest                                   # ~7 minutes on 4 CPU cores

# Phase 1: evolve a species endowment (writes runs/tune/species_endowment.npz)
python -m creature.outer.evolve --preset tune --generations 40 --out runs/tune

# Phase 2: birth the one organism and let it live (resumable after a crash)
python -m creature.handoff --preset tune --runs runs/tune --life runs/life --steps 200000
```

`--preset full` is the SPEC configuration.

## Running at full scale

**Phase 1f (full evolution)** is ~10¹² organism-steps: 4,000 generations ×
513 genomes × 8 lives of 200k steps. Run it on an NVIDIA GPU machine:

```bash
pip install -e ".[gpu]"
python -m creature.outer.evolve --preset full --curriculum 0.015:1000,0.03:1000,0.05:1000,0.07:1000 --out runs/full
```

It checkpoints after every generation (`runs/full/evolve_full_state.npz`), so
it can be stopped and restarted with the same command; it resumes exactly.
Watch `growth_ratio` in `runs/full/evolve_full.csv`: the target band is
0.40–0.60 (SPEC 9.3). Only start the full run once a `tune` run lands there.

**Phase 2 (the life)** must run on a machine that stays up for weeks. Start
it once; if the process or machine dies, run the same command again and the
life resumes from its last checkpoint at the same timestep:

```bash
python -m creature.handoff --preset full --runs runs/full --life runs/life
```

Back up `runs/life/` (the checkpoint and telemetry are the organism).

## Status

See the latest results at the end of [DECISIONS.md](DECISIONS.md). The code
and all ten invariants are complete; fitness tuning (Phase 1e) is in
progress at reduced scale; Phases 1f and 2 need the hardware above.
