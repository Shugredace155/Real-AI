"""Acceptance tests for creature/agent/selfmodel.py (SPEC 7.11)."""

import jax
import jax.numpy as jnp
from _small import SMALL

from creature import organism as O
from creature.agent import selfmodel as sm
from creature.body import sandbox as sb

CFG = SMALL
G = O.default_endowment(jax.random.key(0), CFG)
LIFE = jax.random.key(1)


def _live(org, world, steps, push_every: int = 0, push: float = 0.0):
    """Live `steps` steps; every `push_every` steps an external force shoves the body."""

    def body(c, _):
        o, w = c
        shove = (push_every > 0) & (o.t % max(push_every, 1) == 0)
        k = jax.random.fold_in(LIFE, o.t + 10**7)
        kick = push * jax.random.normal(k, (2,)) * shove
        w = w.replace(agent_vel=w.agent_vel + kick)
        o, w, info = O.step(o, w, O.step_key(LIFE, o.t), CFG)
        return (o, w), (info["self_err"], shove)

    return jax.jit(lambda o, w: jax.lax.scan(body, (o, w), None, length=steps))(org, world)


def test_self_err_decreases() -> None:
    org, world = O.spawn(G, CFG), sb.randomize_world(jax.random.key(2), CFG)
    _, (err, _) = _live(org, world, 100_000)
    early, late = float(err[:10_000].mean()), float(err[-10_000:].mean())
    print(f"self-surprise {early:.4f} -> {late:.4f}")
    assert late < 0.5 * early


def test_distinguishes_agency() -> None:
    # After learning its own body, the organism is more surprised by an external
    # shove than by the consequences of its own actions.
    org, world = O.spawn(G, CFG), sb.randomize_world(jax.random.key(3), CFG)
    (org, world), _ = _live(org, world, 50_000)
    _, (err, shoved) = _live(org, world, 20_000, push_every=97, push=0.5)
    pushed, free = float(err[shoved].mean()), float(err[~shoved].mean())
    print(f"self-surprise when shoved {pushed:.4f}, when acting freely {free:.4f}")
    assert pushed > 2 * free


def test_self_model_is_lived() -> None:
    # Born knowing nothing about itself: the self-model is not part of the endowment.
    assert not any("self" in f for f in O.Endowment.__dataclass_fields__)
    org = O.spawn(G, CFG)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(org.selfm))


def test_vmappable() -> None:
    b = 1024
    n_in, n_s = sm.self_input_dim(CFG), sm.self_state_dim(CFG)
    p = sm.init_selfmodel(CFG)
    s = jax.random.normal(jax.random.key(4), (b, n_in))
    err = jax.vmap(lambda s: sm.self_model(p, s, jnp.zeros(CFG.act_dim), jnp.zeros(n_s)))(s)
    assert err.shape == (b,)
