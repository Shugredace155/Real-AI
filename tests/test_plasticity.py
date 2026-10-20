"""Acceptance tests for creature/substrate/plasticity.py (SPEC 7.5)."""

import jax
import jax.numpy as jnp
import numpy as np

from creature.config import Config
from creature.substrate import neuromod as nmod
from creature.substrate import plasticity as pl

CFG = Config()
# DA = ACh = 1, NE = CORT = 0: gate = 1, so these tests see the raw rule.
NM_ON = jnp.array([1.0, 0.5, 0.0, 1.0, 0.0, 0.0])
NO_META = jnp.inf  # meta_tau = inf disables hardening


def test_oja_bounded() -> None:
    n = 8
    w0 = jnp.zeros((n, n))
    log_lr = jnp.full_like(w0, CFG.init_log_lr)

    def run(meta_tau):
        def body(c, k):
            w, meta, peak = c
            kp, kq = jax.random.split(k)
            pre, post = jax.random.normal(kp, (n,)), jax.random.normal(kq, (n,))
            w, meta = pl.plastic_update(w, log_lr, meta, pre, post, NM_ON, meta_tau, CFG)
            return (w, meta, jnp.maximum(peak, jnp.linalg.norm(w))), None

        keys = jax.random.split(jax.random.key(0), 1_000_000)
        (w, _, peak), _ = jax.jit(lambda ks: jax.lax.scan(body, (w0, jnp.zeros_like(w0), jnp.zeros(())), ks))(keys)
        return float(peak)

    # Without hardening is the worst case; with it, plasticity only shrinks.
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
        new_w, meta = pl.plastic_update(w, log_lr, meta, pre, post, NM_ON, CFG.init_meta_tau, CFG)
        steps.append(float(jnp.sum(jnp.abs(new_w - w))))
        lrs.append(float(pl.effective_lr(log_lr, meta).mean()))
        w = new_w
    assert all(b < a for a, b in zip(steps, steps[1:])), "|dw| not monotonically decreasing"
    assert all(b < a for a, b in zip(lrs, lrs[1:])), "effective learning rate did not harden"


def test_neuromod_gates() -> None:
    nm = NM_ON.at[nmod.DA].set(0.0)
    k0, k1, k2 = jax.random.split(jax.random.key(2), 3)
    w0 = jax.random.normal(k0, (8, 8))
    pre, post = jax.random.normal(k1, (1000, 8)), jax.random.normal(k2, (1000, 8))
    log_lr = jnp.full_like(w0, CFG.init_log_lr)

    def body(c, x):
        w, meta = c
        return pl.plastic_update(w, log_lr, meta, x[0], x[1], nm, CFG.init_meta_tau, CFG), None

    (w, meta), _ = jax.lax.scan(body, (w0, jnp.zeros_like(w0)), (pre, post))
    assert np.array_equal(np.asarray(w), np.asarray(w0))
    assert float(jnp.abs(meta).max()) == 0.0  # a gated-off synapse does not harden


def test_meta_reopens() -> None:
    # An unused synapse's hardening decays monotonically back toward 0.
    meta = jnp.full((4,), 5.0)
    w = jnp.zeros(4)
    history = []
    for _ in range(200):
        w, meta = pl.apply_delta(w, jnp.zeros(4), meta, jnp.zeros(4), NM_ON, CFG.init_meta_tau, CFG)
        history.append(float(meta[0]))
    assert all(0.0 < b < a for a, b in zip(history, history[1:]))


# --- forgetting resistance -------------------------------------------------
# Error-driven delta rule through apply_delta (the rule the PCN uses).
# Task A: input [a, 0] → M a.  Task B: input [c, c] → −M c, directly
# contradicting A on the shared half. B has many solutions; without
# metaplasticity the error splits evenly across A's synapses and the fresh
# ones, cancelling A. With it, A's hardened synapses defer and B is learned on
# the fresh half.
H, NY = 8, 4


def _forgetting_run(meta_tau: float) -> tuple[float, float, float]:
    MA = jax.random.normal(jax.random.key(3), (NY, H)) / jnp.sqrt(H)
    MB = -MA
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
            return pl.apply_delta(w, log_lr, meta, jnp.outer(err, x), NM_ON, meta_tau, CFG), None

        return jax.lax.scan(body, (w, meta), jax.random.split(key, 20_000))[0]

    w = jnp.zeros((NY, 2 * H))
    meta = jnp.zeros_like(w)
    a0, b0 = mse(w, xa, MA), mse(w, xb, MB)
    w, meta = train(w, meta, xa, MA, jax.random.key(4))
    perf_a = 1.0 - mse(w, xa, MA) / a0
    w, meta = train(w, meta, xb, MB, jax.random.key(5))
    return perf_a, 1.0 - mse(w, xa, MA) / a0, 1.0 - mse(w, xb, MB) / b0


def test_forgetting_resistance() -> None:
    post_a, a_after_b, b = _forgetting_run(CFG.init_meta_tau)
    retention = a_after_b / post_a
    print(f"post-A {post_a:.3f}, A after B {a_after_b:.3f} (retention {retention:.2f}), B {b:.3f}")
    assert post_a > 0.95
    assert retention >= 0.70
    assert b > 0.95  # stability did not cost the ability to learn B


def test_forgetting_needs_metaplasticity() -> None:
    # Control: the same run without hardening forgets more.
    post_a, a_after_b, _ = _forgetting_run(NO_META)
    _, a_after_b_meta, _ = _forgetting_run(CFG.init_meta_tau)
    assert a_after_b / post_a < 0.2
    assert a_after_b_meta > a_after_b


def test_vmappable() -> None:
    b = 1024
    k1, k2, k3 = jax.random.split(jax.random.key(6), 3)
    w = jax.random.normal(k1, (b, 8, 8))
    pre, post = jax.random.normal(k2, (b, 8)), jax.random.normal(k3, (b, 8))
    log_lr = jnp.full_like(w, CFG.init_log_lr)
    nm = jnp.broadcast_to(NM_ON, (b, CFG.n_neuromod))
    w2, meta = jax.vmap(lambda w, l, m, p, q, n: pl.plastic_update(w, l, m, p, q, n, CFG.init_meta_tau, CFG))(
        w, log_lr, jnp.zeros_like(w), pre, post, nm
    )
    assert w2.shape == w.shape and meta.shape == w.shape
