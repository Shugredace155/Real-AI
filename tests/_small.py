"""A shrunken Config for whole-organism tests, so long lives fit in CI time.

Same mechanisms, fewer units: narrow PCN, few rays, small episodic buffer,
short sleep and developmental periods.
"""

import dataclasses

from creature.config import Config

SMALL = dataclasses.replace(
    Config(),
    n_rays=8,
    layer_sizes=(16, 16),
    pcn_inference_steps=2,
    episodic_capacity=512,
    dev_episodic_capacity=256,
    sleep_period=2_000,
    sleep_replay_batch=32,
    sleep_consolidation_steps=10,
    dev_steps=4_000,
    fitness_bin_steps=500,
    pop_size=8,
    n_eval_worlds=2,
    growth_window=1_000,
)
