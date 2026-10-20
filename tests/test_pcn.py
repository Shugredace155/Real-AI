"""Acceptance tests for creature/substrate/pcn.py (SPEC 7.4)."""

import ast
import dataclasses
import functools
from pathlib import Path

import jax
import jax.numpy as jnp

from creature.config import Config
from creature.substrate import pcn

CFG = Config()
LINEAR = dataclasses.replace(CFG, pcn_nonlinearity="linear")
SUBSTRATE = Path(pcn.__file__).resolve().parent
GRADIENT_APIS = {"grad", "value_and_grad", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linear_transpose"}

# --- 3-layer linear regression: PCN vs backprop -----------------------------
NY, NH, NX = 4, 16, 8
ETA = 0.01  # per-step learning rate for both learners
NOISE = 0.1


def _regression_setup():
    kA, kd, ki, kt = jax.random.split(jax.random.key(0), 4)
    A = jax.random.normal(kA, (NY, NX)) / jnp.sqrt(NX)

    def data(key, n):
        kx, kn = jax.random.split(key)
        x = jax.random.normal(kx, (n, NX))
        return x, x @ A.T + NOISE * jax.random.normal(kn, (n, NY))

    params, state = pcn.init_pcn(ki, (NY, NH, NX), LINEAR)
    return data(kd, 5000), data(kt, 2000), params, state


def _feedforward(p: pcn.PCNParams, x: jnp.ndarray) -> jnp.ndarray:
    h = p.W_down[1] @ x + p.b[1]
    return p.W_down[0] @ h + p.b[0]


def _test_loss(p, test) -> float:
    X, Y = test
    return float(jnp.mean(jnp.sum((Y - jax.vmap(lambda x: _feedforward(p, x))(X)) ** 2, -1)))


def _train_pcn(params, state, train, n_steps):
    # Top layer clamped to the input, sensory layer to the target (Whittington & Bogacz).
    lr = ETA / LINEAR.dt

    def step(p, xy, n):
        x, y = xy
        out, st = pcn.pcn_forward(p, state, y, jnp.array(1.0), LINEAR, clamp_top=x, n_steps=n)
        return pcn.pcn_update(p, st, out.errors, lr, jnp.array(1.0), LINEAR), None

    return jax.lax.scan(functools.partial(step, n=n_steps), params, train)[0]


def _train_backprop(params, train):
    # The same architecture, loss, learning rate and weight decay, trained by gradient descent.
    decay = ETA * LINEAR.pcn_weight_decay

    def step(p, xy):
        x, y = xy
        g = jax.grad(lambda q: 0.5 * jnp.sum((y - _feedforward(q, x)) ** 2))(p)
        return p.replace(
            W_down=tuple(w - ETA * gw - decay * w for w, gw in zip(p.W_down, g.W_down)),
            b=tuple(b - ETA * gb for b, gb in zip(p.b, g.b)),
        ), None

    return jax.lax.scan(step, params, train)[0]


def test_pcn_approximates_backprop() -> None:
    train, test, params, state = _regression_setup()
    initial = _test_loss(params, test)
    pc = _test_loss(_train_pcn(params, state, train, LINEAR.pcn_inference_steps), test)
    bp = _test_loss(_train_backprop(params, train), test)
    print(f"test loss: initial {initial:.3f}, PCN {pc:.4f}, backprop {bp:.4f}, noise floor {NY * NOISE**2:.4f}")
    assert pc <= 1.05 * bp
    assert pc < 0.01 * initial


def test_relaxation_assigns_credit() -> None:
    # With zero relaxation steps only the bottom layer learns; relaxation is
    # what carries credit to the hidden layer.
    train, test, params, state = _regression_setup()
    none = _test_loss(_train_pcn(params, state, train, 0), test)
    relaxed = _test_loss(_train_pcn(params, state, train, LINEAR.pcn_inference_steps), test)
    assert relaxed < 0.7 * none


def test_no_backprop() -> None:
    for name in ("pcn.py", "plasticity.py", "neuromod.py"):
        tree = ast.parse((SUBSTRATE / name).read_text())
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not used & GRADIENT_APIS, f"{name} uses {used & GRADIENT_APIS}"


def test_error_shapes() -> None:
    sizes = pcn.pcn_sizes(77, CFG)
    params, state = pcn.init_pcn(jax.random.key(1), sizes, CFG)
    out, st = pcn.pcn_forward(params, state, jnp.ones(sizes[0]), jnp.array(1.0), CFG)
    assert tuple(e.shape for e in out.errors) == tuple((n,) for n in sizes[:-1])
    assert tuple(a.shape for a in st.act) == tuple((n,) for n in sizes)
    assert out.prediction.shape == (sizes[0],)
    assert out.pred_err.shape == ()
    d = pcn.pcn_deltas(params, st, out.errors, jnp.array(1.0), CFG)
    assert jax.tree.map(jnp.shape, d) == jax.tree.map(jnp.shape, params)


def test_learns_temporal_sequence() -> None:
    # A moving signal (a rotating phase) is only predictable through lateral,
    # temporal context; prior prediction error must fall as the PCN learns it.
    n = 16
    sizes = (n, 32, 16)
    params, state = pcn.init_pcn(jax.random.key(2), sizes, CFG)
    lr = jnp.exp(CFG.init_log_lr)
    phases = jnp.arange(n) * 2 * jnp.pi / n

    def body(c, t):
        p, s = c
        x = jnp.sin(phases + 0.01 * t)
        out, s = pcn.pcn_forward(p, s, x, jnp.array(1.0), CFG)
        return (pcn.pcn_update(p, s, out.errors, lr, jnp.array(1.0), CFG), s), out.pred_err

    _, errs = jax.lax.scan(body, (params, state), jnp.arange(50_000.0))
    early, late = float(errs[:1000].mean()), float(errs[-1000:].mean())
    assert late < 0.1 * early, (early, late)


def test_w_up_tracks_transpose() -> None:
    # W_up and W_down receive transposed updates and share weight decay, so
    # any mismatch shrinks by exactly (1 − dt·lr·decay) per step (Kolen–Pollack).
    sizes = (12, 10, 6)
    params, state = pcn.init_pcn(jax.random.key(3), sizes, CFG)
    params = params.replace(W_up=tuple(w + 0.5 for w in params.W_up))
    lr = jnp.exp(CFG.init_log_lr)
    mismatch = lambda p: float(sum(jnp.sum((u - d.T) ** 2) for u, d in zip(p.W_up, p.W_down)) ** 0.5)
    before = mismatch(params)
    steps = 2000

    def body(c, k):
        p, s = c
        out, s = pcn.pcn_forward(p, s, jax.random.normal(k, (sizes[0],)), jnp.array(1.0), CFG)
        return (pcn.pcn_update(p, s, out.errors, lr, jnp.array(1.0), CFG), s), None

    (params, _), _ = jax.lax.scan(body, (params, state), jax.random.split(jax.random.key(4), steps))
    expected = before * (1.0 - CFG.dt * float(lr) * CFG.pcn_weight_decay) ** steps
    assert abs(mismatch(params) - expected) < 1e-3 * before


def test_vmappable() -> None:
    b = 1024
    sizes = pcn.pcn_sizes(77, CFG)
    params, state = jax.vmap(lambda k: pcn.init_pcn(k, sizes, CFG))(jax.random.split(jax.random.key(5), b))
    x = jnp.ones((b, sizes[0]))
    out, st = jax.vmap(lambda p, s, x: pcn.pcn_forward(p, s, x, jnp.array(1.0), CFG))(params, state, x)
    assert out.pred_err.shape == (b,)
    assert tuple(e.shape for e in out.errors) == tuple((b, n) for n in sizes[:-1])
    p2 = jax.vmap(lambda p, s, e: pcn.pcn_update(p, s, e, jnp.array(1.0), jnp.array(1.0), CFG))(params, st, out.errors)
    assert jax.tree.map(jnp.shape, p2) == jax.tree.map(jnp.shape, params)
