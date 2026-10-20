"""Shared Invariant I5 check: every neuromodulator channel moves at least one knob.

A channel is causal if clamping it to 0 vs NM_MAX (others held at 1.0) changes
some entry of `neuromod.knobs` by more than 1e-3. Knobs are consumed by the
PCN (precision, plasticity_rate), the policy (temperature, discount,
social_weight) and episodic memory (encoding). For the channels that act on
the policy (NE, 5HT, OT) the check also requires a change in the action
distribution, i.e. in behavior.
"""

import jax
import jax.numpy as jnp

from creature.config import Config
from creature.substrate import neuromod as nmod

CFG = Config()
TOL = 1e-3


# Channels whose knobs act on the policy must also change the action distribution.
BEHAVIORAL = (nmod.NE, nmod.SHT, nmod.OT)


def _action_probs(nm: jnp.ndarray) -> jnp.ndarray:
    from creature import organism as O
    from creature.agent import policy as pol

    g = O.default_endowment(jax.random.key(0), CFG)
    n_out, n_feat = pol.dyn_sizes(CFG)
    W = 0.3 * jax.random.normal(jax.random.key(1), (n_out, n_feat))
    W = W.at[pol.z_dim(CFG) + CFG.n_drives].set(jnp.abs(W[pol.z_dim(CFG) + CFG.n_drives]))  # some actions draw the caregiver
    dyn = pol.DynParams(W=W, b=jnp.zeros(n_out))
    z = jax.random.normal(jax.random.key(2), (pol.z_dim(CFG),))
    err = jnp.full(CFG.n_drives, 0.2)
    return jax.nn.softmax(pol.policy_logits(dyn, z, err, nmod.knobs(nm), g, CFG))


def check_channel_causal(channel: int) -> None:
    base = jnp.full(CFG.n_neuromod, 1.0)
    lo_nm, hi_nm = base.at[channel].set(0.0), base.at[channel].set(nmod.NM_MAX)
    lo, hi = nmod.knobs(lo_nm), nmod.knobs(hi_nm)
    moved = [k for k in lo if float(jnp.abs(lo[k] - hi[k])) > TOL]
    assert moved, f"{nmod.CHANNELS[channel]} moves no knob"
    if channel in BEHAVIORAL:
        shift = float(jnp.abs(_action_probs(lo_nm) - _action_probs(hi_nm)).max())
        assert shift > TOL, f"{nmod.CHANNELS[channel]} does not change behavior"
