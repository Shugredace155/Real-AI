"""Acceptance tests for creature/substrate/plasticity.py (SPEC 7.5, DECISIONS D31)."""

import jax
import jax.numpy as jnp
import numpy as np
from _birth import birth_genome

from creature.config import Config
from creature.substrate import neuromod as nmod
from creature.substrate import plasticity as pl

CFG = Config()
G = birth_genome()
BIRTH_GATE = nmod.plasticity_gate(nmod.init_neuromod(G).nm)
MAX_GATE = nmod.plasticity_gate(jnp.array([2.0, 0.0, 2.0, 2.0, 0.0, 0.0]))  # DA, NE, ACh at max, CORT 0
NO_META = jnp.inf  # meta_tau = inf disables hardening


def test_oja_bounded() -> None:
    n = 8
    w0 = jnp.zeros((n, n))
    log_lr = jnp.full_like(w0, CFG.init_log_lr)

    def run(meta_tau):
        g = birth_genome(meta_tau=meta_tau)

        def body(c, k):
            w, meta, peak = c
            kp, kq = jax.random.split(k)
            pre, post = jax.random.normal(kp, (n,)), jax.random.normal(kq, (n,))
            w, meta = pl.plastic_update(w, log_lr, meta, pre, post, MAX_GATE, g, CFG.dt)
            return (w, meta, jnp.maximum(peak, jnp.linalg.norm(w))), None

        keys = jax.random.split(jax.random.key(0), 1_000_000)
        (_, _, peak), _ = jax.jit(lambda ks: jax.lax.scan(body, (w0, jnp.zeros_like(w0), jnp.zeros(())), ks))(keys)
        return float(peak)

    # Worst case: the largest possible gate and no hardening.
    for tau in (NO_META, CFG.init_meta_tau):
        peak = run(tau)
        assert np.isfinite(peak) and peak < 10.0, peak


def test_metaplasticity_hardens() -> None:
    k1, k2 = jax.random.split(jax.random.key(1))
    pre, post = jax.random.normal(k1, (8,)), jax.random.normal(k2, (8,))
    w = jnp.zeros((8, 8))
    meta = jnp.zeros_like(w)
    log_lr = jnp.full_like(w, CFG.init_log_lr)
    steps, lrs = [], []
    for _ in range(500):
        new_w, meta = pl.plastic_update(w, log_lr, meta, pre, post, BIRTH_GATE, G, CFG.dt)
        steps.append(float(jnp.sum(jnp.abs(new_w - w))))
        lrs.append(float(pl.effective_lr(log_lr, meta).mean()))
        w = new_w
    assert all(b < a for a, b in zip(steps, steps[1:])), "|dw| not monotonically decreasing"
    assert all(b < a for a, b in zip(lrs, lrs[1:])), "effective learning rate did not harden"


def test_gate_zero_freezes() -> None:
    k0, k1, k2, k3 = jax.random.split(jax.random.key(2), 4)
    w0 = jax.random.normal(k0, (8, 8))
    meta0 = jnp.abs(jax.random.normal(k3, (8, 8)))
    pre, post = jax.random.normal(k1, (1000, 8)), jax.random.normal(k2, (1000, 8))
    log_lr = jnp.full_like(w0, CFG.init_log_lr)

    def body(c, x):
        w, meta = c
        w, new_meta = pl.plastic_update(w, log_lr, meta, x[0], x[1], jnp.array(0.0), G, CFG.dt)
        return (w, new_meta), jnp.max(new_meta - meta)

    (w, _), rises = jax.lax.scan(body, (w0, meta0), (pre, post))
    assert np.array_equal(np.asarray(w), np.asarray(w0))
    # No hardening without plasticity; passive decay (reopening) still runs by design.
    assert float(rises.max()) <= 0.0


def test_neuromod_gates() -> None:
    for nm in (nmod.init_neuromod(G).nm, jnp.ones(CFG.n_neuromod)):
        full = nmod.plasticity_gate(nm)
        no_da = nmod.plasticity_gate(nm.at[nmod.DA].set(0.0))
        assert float(no_da) <= 0.6 * float(full)


def test_meta_reopens() -> None:
    # An unused synapse's hardening decays monotonically back toward 0.
    meta = jnp.full((4,), 5.0)
    w = jnp.zeros(4)
    history = []
    for _ in range(200):
        w, meta = pl.apply_delta(w, jnp.zeros(4), meta, jnp.zeros(4), BIRTH_GATE, CFG.init_meta_tau, CFG.dt)
        history.append(float(meta[0]))
    assert all(0.0 < b < a for a, b in zip(history, history[1:]))


# --- forgetting resistance -------------------------------------------------
# Error-driven delta rule through apply_delta (the rule the PCN uses).
# Task A: input [a, 0] → M a.  Task B: input [c, c] → −M c, directly
# contradicting A on the shared half. B has many solutions; without
# metaplasticity the error splits evenly across A's synapses and the fresh
# ones, cancelling A. With it, A's hardened synapses defer to the fresh half.
H, NY = 8, 4
STEPS = 10_000


def _forgetting_run(meta_tau: float) -> dict[str, float]:
    MA = jax.random.normal(jax.random.key(3), (NY, H)) / jnp.sqrt(H)
    xa = lambda a: jnp.concatenate([a, jnp.zeros(H)])
    xb = lambda c: jnp.concatenate([c, c])
    test_in = jax.random.normal(jax.random.key(9), (1000, H))

    def mse(w, mk, M):
        X = jax.vmap(mk)(test_in)
        return float(jnp.mean(jnp.sum((test_in @ M.T - X @ w.T) ** 2, -1)))

    def train(w, meta, mk, M, key):
        log_lr = jnp.full_like(w, CFG.init_log_lr)

        def body(c, k):
            w, meta = c
            a = jax.random.normal(k, (H,))
            x = mk(a)
            err = M @ a - w @ x
            return pl.apply_delta(w, log_lr, meta, jnp.outer(err, x), BIRTH_GATE, meta_tau, CFG.dt), None

        return jax.lax.scan(body, (w, meta), jax.random.split(key, STEPS))[0]

    w = jnp.zeros((NY, 2 * H))
    meta = jnp.zeros_like(w)
    initial = mse(w, xa, MA)
    w, meta = train(w, meta, xa, MA, jax.random.key(4))
    post_a = mse(w, xa, MA)
    w, meta = train(w, meta, xb, -MA, jax.random.key(5))
    return dict(initial=initial, post_a=post_a, a_after_b=mse(w, xa, MA), b=mse(w, xb, -MA))


def test_forgetting_resistance() -> None:
    # Retention of improvement: the share of A's error reduction that survives B.
    r = _forgetting_run(CFG.init_meta_tau)
    gained = r["initial"] - r["post_a"]
    learned = gained / r["initial"]
    retention = (r["initial"] - r["a_after_b"]) / gained
    print(
        f"A error: initial {r['initial']:.3f}, post-A {r['post_a']:.4f}, after B {r['a_after_b']:.4f}; "
        f"A learned {learned:.2f}, retention {retention:.2f}; B error {r['b']:.4f}"
    )
    # Guard: a synapse that barely learns A trivially "retains" it.
    assert learned >= 0.5, f"task A was not learned (only {learned:.0%} of its error removed)"
    assert retention >= 0.70


def test_forgetting_needs_metaplasticity() -> None:
    # Control: the same run without hardening forgets more.
    assert _forgetting_run(CFG.init_meta_tau)["a_after_b"] < _forgetting_run(NO_META)["a_after_b"]


def test_vmappable() -> None:
    b = 1024
    k1, k2, k3 = jax.random.split(jax.random.key(6), 3)
    w = jax.random.normal(k1, (b, 8, 8))
    pre, post = jax.random.normal(k2, (b, 8)), jax.random.normal(k3, (b, 8))
    log_lr = jnp.full_like(w, CFG.init_log_lr)
    gate = jnp.full((b,), BIRTH_GATE)
    w2, meta = jax.vmap(lambda w, l, m, p, q, gt: pl.plastic_update(w, l, m, p, q, gt, G, CFG.dt))(
        w, log_lr, jnp.zeros_like(w), pre, post, gate
    )
    assert w2.shape == w.shape and meta.shape == w.shape
