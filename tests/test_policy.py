"""Acceptance tests for creature/agent/policy.py (SPEC 7.10)."""

import jax
import jax.numpy as jnp
from _small import SMALL

from creature import organism as O
from creature.agent import policy as pol
from creature.substrate import neuromod as nmod

CFG = SMALL
A = pol.n_actions(CFG)
G = O.default_endowment(jax.random.key(0), CFG)
Z = jnp.zeros(pol.z_dim(CFG))
E = jnp.full(CFG.n_drives, 0.2)
_, N_FEAT = pol.dyn_sizes(CFG)
ZD = pol.z_dim(CFG)


def _dyn_with(action_de: jnp.ndarray | None = None, action_prox: jnp.ndarray | None = None) -> pol.DynParams:
    """A forward model whose predictions depend only on the action one-hot."""
    dyn = pol.init_dyn(CFG)
    W = dyn.W
    onehot_cols = slice(ZD, ZD + A)
    if action_de is not None:  # every drive error changes by action_de[a] per period
        W = W.at[ZD : ZD + CFG.n_drives, onehot_cols].set(jnp.broadcast_to(action_de, (CFG.n_drives, A)))
    if action_prox is not None:
        W = W.at[ZD + CFG.n_drives, onehot_cols].set(action_prox)
    return dyn.replace(W=W)


def _knobs(**levels) -> dict:
    nm = jnp.ones(CFG.n_neuromod)
    for name, v in levels.items():
        nm = nm.at[nmod.CHANNELS.index(name)].set(v)
    return nmod.knobs(nm)


def _probs(dyn, knobs, g=G) -> jnp.ndarray:
    return jax.nn.softmax(pol.policy_logits(dyn, Z, E, knobs, g, CFG))


def test_ne_gates_exploration() -> None:
    dyn = _dyn_with(action_de=-jnp.linspace(0.0, 1.0, A))  # last action reduces error most
    h_low = float(pol.entropy(jnp.log(_probs(dyn, _knobs(NE=0.0)))))
    h_high = float(pol.entropy(jnp.log(_probs(dyn, _knobs(NE=2.0)))))
    assert h_low < 0.1 * jnp.log(A), "low NE should be near-deterministic"
    assert h_high > 0.4 * jnp.log(A), "high NE should explore"
    assert h_high > 10 * h_low
    G_ = pol.expected_free_energy(dyn, Z, E, _knobs(NE=0.0), G, CFG)
    assert int(jnp.argmax(_probs(dyn, _knobs(NE=0.0)))) == int(jnp.argmin(G_))  # exploits the best rollout


def test_ot_gates_social() -> None:
    voc = 9  # the vocalize primitive brings the caregiver closer...
    prox = jnp.zeros(A).at[voc].set(1.0)
    de = jnp.zeros(A).at[voc].set(0.02)  # ...at a small homeostatic cost
    dyn = _dyn_with(action_de=de, action_prox=prox)
    p_low = float(_probs(dyn, _knobs(OT=0.0))[voc])
    p_high = float(_probs(dyn, _knobs(OT=2.0))[voc])
    assert p_high > 2 * p_low


def test_sht_gates_discount() -> None:
    # Far-sighted (high 5HT) and myopic (low 5HT) organisms weigh the horizon differently.
    dyn = pol.DynParams(W=0.3 * jax.random.normal(jax.random.key(1), pol.dyn_sizes(CFG)), b=jnp.zeros(pol.dyn_sizes(CFG)[0]))
    z = jax.random.normal(jax.random.key(2), (ZD,))
    p0 = jax.nn.softmax(pol.policy_logits(dyn, z, E, _knobs(**{"5HT": 0.0}), G, CFG))
    p2 = jax.nn.softmax(pol.policy_logits(dyn, z, E, _knobs(**{"5HT": 2.0}), G, CFG))
    assert float(jnp.abs(p0 - p2).max()) > 1e-3


def test_reflex_biases_action() -> None:
    # An inherited reflex: hunger (energy error) → grab. Works with no learned model at all.
    grab = 7
    reflex = jnp.zeros(pol.n_reflex(CFG)).at[A + grab * CFG.n_drives + 0].set(-20.0)  # negative error = hungry
    g = G.replace(reflex=reflex)
    hungry = jnp.zeros(CFG.n_drives).at[0].set(-0.5)
    logits = pol.policy_logits(pol.init_dyn(CFG), Z, hungry, _knobs(), g, CFG)
    assert int(jnp.argmax(logits)) == grab


def test_forward_model_learns() -> None:
    # Local delta rule on synthetic transitions with an action-dependent outcome.
    true = 0.5 * jax.random.normal(jax.random.key(3), pol.dyn_sizes(CFG))
    dyn = pol.init_dyn(CFG)

    def body(d, k):
        kz, ka = jax.random.split(k)
        z, a = jax.random.normal(kz, (ZD,)), jax.random.randint(ka, (), 0, A)
        target = true @ pol.dyn_features(z, a, CFG)
        delta = pol.dyn_deltas(d, z, a, target, CFG)
        new = jax.tree.map(lambda w, g: w + 0.01 * g, d, delta)
        return new, jnp.mean((target - (d.W @ pol.dyn_features(z, a, CFG) + d.b)) ** 2)

    _, err = jax.lax.scan(body, dyn, jax.random.split(jax.random.key(4), 20_000))
    assert float(err[-500:].mean()) < 0.05 * float(err[:500].mean())


def test_policy_vmappable() -> None:
    b = 1024
    zs = jax.random.normal(jax.random.key(5), (b, ZD))
    keys = jax.random.split(jax.random.key(6), b)
    kn = _knobs()
    a, logits = jax.vmap(lambda z, k: pol.policy(pol.init_dyn(CFG), z, E, kn, G, CFG, k))(zs, keys)
    assert a.shape == (b,) and logits.shape == (b, A)
