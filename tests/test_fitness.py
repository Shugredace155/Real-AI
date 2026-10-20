"""Acceptance tests for creature/outer/fitness.py (SPEC 7.13, 9)."""

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature import life
from creature import organism as O
from creature.body import sandbox as sb
from creature.outer import fitness as F
from creature.outer import genome as G

CFG = SMALL
COMPONENTS = jax.jit(lambda g, k: F.fitness_components(g, k, CFG))


def test_fitness_deterministic() -> None:
    g = G.init_genome(jax.random.key(0), CFG)
    a = COMPONENTS(g, jax.random.key(1))
    b = COMPONENTS(g, jax.random.key(1))
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_fitness_varies() -> None:
    base = G.init_genome(jax.random.key(2), CFG).flat()
    fits = []
    for i in range(8):
        noise = 0.3 * jax.random.normal(jax.random.key(100 + i), base.shape)
        fits.append(float(COMPONENTS(G.Genome.from_flat(base + noise, CFG), jax.random.key(3))["fitness"]))
    print("random-genome fitness:", np.round(fits, 3))
    assert np.std(fits) >= 0.1


def _born_competent_rigid(adaptive: G.Genome) -> G.Genome:
    """Inherit the weights an organism learned over a life, but with plasticity switched off."""
    e = G.express(adaptive, CFG)
    org, world = O.spawn(e, CFG), sb.randomize_world(jax.random.key(4), CFG)
    org, _, _, _ = jax.jit(lambda o, w: life.awake(o, w, jax.random.key(5), 40, 500, CFG))(org, world)
    w, _ = jax.flatten_util.ravel_pytree(G.Inherited(org.pcn.W_down, org.pcn.W_lat, org.pcn.b, org.dyn))
    return adaptive.replace(init_w=w, log_lr=jnp.full_like(adaptive.log_lr, -12.0))


def test_learn_speed_dominates() -> None:
    adaptive = G.init_genome(jax.random.key(6), CFG)
    rigid = _born_competent_rigid(adaptive)
    key = jax.random.key(7)
    a, r = COMPONENTS(adaptive, key), COMPONENTS(rigid, key)
    print("adaptive:", {k: round(float(v), 3) for k, v in a.items()})
    print("rigid:   ", {k: round(float(v), 3) for k, v in r.items()})
    assert float(r["learn_speed"]) < float(a["learn_speed"])
    assert float(a["fitness"]) > float(r["fitness"])


def test_rejects_statue() -> None:
    g = G.init_genome(jax.random.key(8), CFG)
    statue = g.replace(log_lr=jnp.full_like(g.log_lr, -12.0))
    c = COMPONENTS(statue, jax.random.key(9))
    assert float(c["rejected"]) == 1.0 and float(c["fitness"]) == CFG.reject_fitness


def test_rejects_trivial_setpoints() -> None:
    g = G.init_genome(jax.random.key(10), CFG)
    trivial = g.replace(setpoint=jnp.asarray(CFG.drive_init))  # already satisfied at birth
    c = COMPONENTS(trivial, jax.random.key(11))
    assert float(c["rejected"]) == 1.0


def test_growth_ratio_range() -> None:
    xs = jnp.linspace(0.0, 1.0, 11)
    gr = jax.vmap(lambda b: jax.vmap(lambda e: F.growth_ratio(b, e))(xs))(xs)
    assert bool(jnp.all((gr >= -1) & (gr <= 1)))
    assert float(F.growth_ratio(jnp.array(0.5), jnp.array(1.0))) == 0.5
