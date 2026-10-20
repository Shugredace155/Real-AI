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

`--preset full` is the SPEC configuration. Phase 1f at full scale (4,000
generations × 512 genomes × 8 lives of 200k steps) needs a GPU and days.
