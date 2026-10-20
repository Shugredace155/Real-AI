"""Acceptance tests for creature/outer/genome.py (SPEC 7.12)."""

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature import organism as O
from creature.config import Config
from creature.outer import genome as G

CFG = Config()


def test_flat_roundtrip() -> None:
    g = G.init_genome(jax.random.key(0), CFG)
    back = G.Genome.from_flat(g.flat(), CFG)
    for f in G.FIELDS:
        assert np.array_equal(np.asarray(getattr(g, f)), np.asarray(getattr(back, f))), f
    assert g.flat().shape == (G.genome_dim(CFG),)


def test_dim_bounded() -> None:
    dim = G.genome_dim(CFG)
    print(f"genome dimension: {dim:,} ({G.n_synapses(CFG):,} inherited synapses × 3 + {dim - 3 * G.n_synapses(CFG)} scalars)")
    assert dim <= 300_000


def test_no_lived_in_genome() -> None:
    lived = ("episodic", "self", "meta", "fast", "attach", "mood", "temperament")
    allowed = {"meta_tau"}  # the hardening *rate* is inherited; the hardened *state* is lived
    for f in G.FIELDS:
        assert f in allowed or not any(word in f for word in lived), f
    # Nothing the organism learns or remembers is expressible from the genome.
    e = G.express(G.init_genome(jax.random.key(1), CFG), CFG)
    assert not any("self" in f or "episodic" in f for f in e.__dataclass_fields__)


def test_express_is_birth() -> None:
    g = G.init_genome(jax.random.key(2), SMALL)
    e = G.express(g, SMALL)
    org = O.spawn(e, SMALL)
    # Every weight at birth comes from the genome (Invariant I1).
    w, _ = jax.flatten_util.ravel_pytree(G.Inherited(org.pcn.W_down, org.pcn.W_lat, org.pcn.b, org.dyn))
    assert np.array_equal(np.asarray(w), np.asarray(g.init_w))
    for up, down in zip(org.pcn.W_up, org.pcn.W_down):
        assert np.array_equal(np.asarray(up), np.asarray(down.T))
    assert float(e.da_tonic[0]) == float(g.da_tonic[0])
    assert np.allclose(np.asarray(e.nm_tau), np.asarray(SMALL.nm_tau_init))


def test_da_tonic_prior() -> None:
    lo, hi = CFG.da_tonic_prior
    v = jax.vmap(lambda k: G.init_genome(k, SMALL).da_tonic[0])(jax.random.split(jax.random.key(3), 500))
    assert float(v.min()) >= lo and float(v.max()) <= hi
    assert float(v.max() - v.min()) > 0.5 * (hi - lo)  # actually spread over the prior


def test_save_load_roundtrip(tmp_path) -> None:
    g = G.init_genome(jax.random.key(4), SMALL)
    path = tmp_path / "genome.npz"
    g.save(path)
    back = G.Genome.load(path, SMALL)
    assert np.array_equal(np.asarray(g.flat()), np.asarray(back.flat()))


def test_constrained_decoding() -> None:
    g = G.init_genome(jax.random.key(5), SMALL)
    wild = G.Genome.from_flat(g.flat() + 50.0 * jax.random.normal(jax.random.key(6), g.flat().shape), SMALL)
    e = G.express(wild, SMALL)
    assert bool(jnp.all((e.setpoint >= 0) & (e.setpoint <= 1)))
    assert bool(jnp.all(e.urgency >= 0)) and bool(jnp.all(e.nm_tau > 0))
    assert all(bool(jnp.all(x > 0)) for x in jax.tree.leaves(e.pcn_meta_tau))
