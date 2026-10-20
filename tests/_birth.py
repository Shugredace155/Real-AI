"""A stand-in for the genome fields the substrate reads, until outer/genome.py exists (step 7).

Holds only the fields used by neuromod, plasticity and affect, at their birth
defaults from Config. `da_tonic` defaults to the midpoint of its prior.
"""

import jax.numpy as jnp
from flax import struct

from creature.config import Config

CFG = Config()


@struct.dataclass
class BirthGenome:
    nm_gain: jnp.ndarray  # (6,)
    nm_tau: jnp.ndarray  # (6,)
    da_tonic: jnp.ndarray  # (1,)
    meta_tau: jnp.ndarray  # () or per-synapse
    setpoint: jnp.ndarray  # (n_drives,)
    urgency: jnp.ndarray  # (n_drives,)


def birth_genome(da_tonic: float | None = None, meta_tau: float | None = None) -> BirthGenome:
    lo, hi = CFG.da_tonic_prior
    return BirthGenome(
        nm_gain=jnp.asarray(CFG.nm_gain_init),
        nm_tau=jnp.asarray(CFG.nm_tau_init),
        da_tonic=jnp.array([(lo + hi) / 2 if da_tonic is None else da_tonic]),
        meta_tau=jnp.asarray(CFG.init_meta_tau if meta_tau is None else meta_tau),
        setpoint=jnp.asarray(CFG.setpoint_init),
        urgency=jnp.asarray(CFG.urgency_init),
    )
