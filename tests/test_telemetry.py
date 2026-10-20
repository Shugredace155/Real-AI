"""Acceptance tests for creature/telemetry.py (SPEC 7.16)."""

import jax
import jax.numpy as jnp
import numpy as np
from _small import SMALL

from creature import organism as O
from creature import telemetry as tm
from creature.body import sandbox as sb

CFG = SMALL


def _rows(steps: int) -> dict:
    org, world = O.spawn(O.default_endowment(jax.random.key(0), CFG), CFG), sb.randomize_world(jax.random.key(1), CFG)

    def body(c, _):
        o, w = c
        o, w, info = O.step(o, w, O.step_key(jax.random.key(2), o.t), CFG)
        return (o, w), tm.row(o, info)

    return jax.jit(lambda o, w: jax.lax.scan(body, (o, w), None, length=steps))(org, world)[1]


def test_telemetry_shapes() -> None:
    r = _rows(50)
    assert set(r) == set(tm.FIELDS)
    expected = {"drives": (50, CFG.n_drives), "nm": (50, CFG.n_neuromod)}
    for k, v in r.items():
        assert v.shape == expected.get(k, (50,)), (k, v.shape)
    assert np.array_equal(np.asarray(r["t"]), np.arange(1, 51))
    assert bool(jnp.all(jnp.isfinite(r["weight_norm"])))


def test_growth_ratio_range() -> None:
    w = CFG.growth_window
    for seed in range(5):
        health = jax.random.uniform(jax.random.key(seed), (5 * w,))
        if seed == 0:
            health = health.at[:w].set(1.0).at[w:].set(0.0)  # competence collapses to 0
        gr = tm.init_growth()
        ratios = []
        for t, h in enumerate(health):
            gr = tm.update_growth(gr, h, jnp.asarray(t), CFG)
            ratios.append(float(gr.ratio))
        ratios = np.asarray(ratios)
        defined = ratios[2 * w - 1 :]
        assert np.all(np.isnan(ratios[: 2 * w - 1])), "undefined until a second window completes"
        assert np.all(np.isfinite(defined)) and np.all((defined >= -1) & (defined <= 1))


def test_growth_ratio_value() -> None:
    w = CFG.growth_window
    health = jnp.concatenate([jnp.full(w, 0.4), jnp.full(w, 0.8)])  # competence doubles
    gr = tm.init_growth()
    for t, h in enumerate(health):
        gr = tm.update_growth(gr, h, jnp.asarray(t), CFG)
    assert abs(float(gr.ratio) - 0.5) < 1e-5


def test_store_roundtrip(tmp_path) -> None:
    store = tm.TelemetryStore(tmp_path / "telemetry")
    r = _rows(30)
    first = {k: v[:10] for k, v in r.items()}
    second = {k: v[10:] for k, v in r.items()}
    store.write(first)
    store.write(second)
    back = store.read()
    for k in r:
        assert np.array_equal(back[k], np.asarray(r[k])), k
