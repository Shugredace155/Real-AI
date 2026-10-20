"""All hyperparameters for PROJECT CREATURE (SPEC 7.1).

Every magic number in the codebase lives here; module bodies take a `Config`
instead of using literals. The dataclass is frozen so it is hashable and can be
passed to `jax.jit` as a static argument without being traced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Frozen global configuration. Fields map one-to-one to SPEC 7.1."""

    dt: float = 1e-3  # simulation timestep in seconds (1 ms real-time clock, SPEC 5.1)
    n_drives: int = 6  # number of homeostatic drives (SPEC 7.2)
    n_neuromod: int = 6  # number of neuromodulator channels: DA, 5HT, NE, ACh, OT, CORT (SPEC 7.6)
    layer_sizes: tuple[int, ...] = (64, 128, 128, 64)  # predictive-coding layer widths, bottom to top (SPEC 7.4)
    n_fast_synapses: int = 0  # fast-weight synapse count; 0 until the Phase 2 addition (SPEC 7.5)
    episodic_capacity: int = 100_000  # episodic ring-buffer slots (SPEC 7.8)
    sleep_period: int = 60_000  # awake steps between sleep episodes (SPEC 7.9)
    sleep_replay_batch: int = 512  # traces replayed per consolidation step (SPEC 7.9)
    sleep_consolidation_steps: int = 200  # consolidation passes per sleep episode (SPEC 7.9)
    pop_size: int = 512  # evolution population size per generation (SPEC 7.14)
    n_eval_worlds: int = 8  # randomized worlds each genome is evaluated in (SPEC 7.13, 9.2)
    dev_steps: int = 200_000  # lifetime steps per developmental fitness rollout (SPEC 7.13)
    generations: int = 4000  # evolution generations in the full Phase 1f run (SPEC 7.14)
    # fitness weights — see 9.1
    w_survival: float = 1.0  # weight on fraction of dev_steps alive (SPEC 9.1)
    w_learn: float = 3.0  # weight on relative early→late pred_err reduction; the anti-statue term (SPEC 9.1)
    w_competence: float = 1.0  # weight on mean task score in the final quarter (SPEC 9.1)
    w_robustness: float = 2.0  # weight on recovery after body perturbation (SPEC 9.1)
    w_curiosity: float = 0.5  # weight on mean intrinsic reward; the anti-plateau term (SPEC 9.1)
    w_hack: float = 0.5  # penalty weight on exploit magnitude, subtracted (SPEC 9.1)
