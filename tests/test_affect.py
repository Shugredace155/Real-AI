"""Acceptance tests for creature/affect/core.py (SPEC 7.7)."""

import ast
import inspect

import jax
import jax.numpy as jnp
import pytest
from _birth import birth_genome

from creature.affect import core as affect
from creature.body import homeostat as hs
from creature.body import sandbox as sb
from creature.config import Config

CFG = Config()
G = birth_genome()
SET = G.setpoint


def _affect_of(drive_path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run step_affect along a (T, n_drives) path; returns per-step valence and arousal."""

    def body(c, d):
        prev_d, prev_e = c
        v, a, e = affect.step_affect(d, prev_d, prev_e, G, CFG.dt)
        return (d, e), (v, a)

    start = drive_path[0]
    _, (v, a) = jax.lax.scan(body, (start, affect.error_magnitude(start, G)), drive_path)
    return v, a


def _path_toward(start: jnp.ndarray, end: jnp.ndarray, steps: int) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, steps)[:, None]
    return start + t * (end - start)


def test_valence_sign() -> None:
    far = jnp.clip(SET - 0.4, 0.0, 1.0)
    v_better, _ = _affect_of(_path_toward(far, SET, 200))
    v_worse, _ = _affect_of(_path_toward(SET, far, 200))
    assert bool(jnp.all(v_better[1:] > 0)), "falling error must feel good"
    assert bool(jnp.all(v_worse[1:] < 0)), "rising error must feel bad"


def test_valence_transient() -> None:
    # Improve, then hold still far from the setpoint: valence returns to exactly
    # 0 even though the error *level* is still large. Pleasure does not persist.
    far = jnp.clip(SET - 0.4, 0.0, 1.0)
    mid = jnp.clip(SET - 0.2, 0.0, 1.0)
    path = jnp.concatenate([_path_toward(far, mid, 100), jnp.broadcast_to(mid, (100, CFG.n_drives))])
    v, _ = _affect_of(path)
    assert float(v[50]) > 0
    assert bool(jnp.all(v[101:] == 0.0))
    assert float(affect.error_magnitude(mid, G)) > 0.5  # not because the creature is satisfied


def test_valence_is_rate_per_second() -> None:
    # Valence is −d(err)/dt in units of 1/s, independent of the timestep size.
    d0 = jnp.clip(SET - 0.3, 0.0, 1.0)
    rate = 0.2  # error falls at this many units per second
    step = jnp.zeros(CFG.n_drives).at[0].set(rate)  # one drive moving toward its setpoint
    for dt in (1e-3, 1e-2):
        v, _, _ = affect.step_affect(d0 + step * dt, d0, affect.error_magnitude(d0, G), G, dt)
        assert float(v) == pytest.approx(rate, rel=1e-2)  # float32 differencing


def test_arousal_bounded() -> None:
    k1, k2 = jax.random.split(jax.random.key(0))
    drives = jax.random.uniform(k1, (100_000, CFG.n_drives))
    prev = jax.random.uniform(k2, (100_000, CFG.n_drives))
    errs = jax.vmap(lambda d: affect.error_magnitude(d, G))(prev)
    _, a, _ = jax.vmap(lambda d, p, e: affect.step_affect(d, p, e, G, CFG.dt))(drives, prev, errs)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_affect_from_body() -> None:
    """I7: valence is a pure function of drives (plus evolved setpoints), with no label input."""
    params = list(inspect.signature(affect.step_affect).parameters)
    assert params == ["drives", "prev_drives", "prev_err_mag", "g", "dt"]
    tree = ast.parse(inspect.getsource(affect))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"jax", "typing", "__future__"}, imported
    # No text-typed input anywhere in the module's public functions.
    for fn in (affect.step_affect, affect.error_magnitude, affect.homeostatic_error):
        for param in inspect.signature(fn).parameters.values():
            assert "str" not in str(param.annotation), f"{fn.__name__}({param.name}) accepts text"
    # Same body, same feeling: repeated calls are bit-identical.
    d, p = jnp.full(CFG.n_drives, 0.3), jnp.full(CFG.n_drives, 0.31)
    e = affect.error_magnitude(p, G)
    a = affect.step_affect(d, p, e, G, CFG.dt)
    b = affect.step_affect(d, p, e, G, CFG.dt)
    assert all(bool(x == y) for x, y in zip(a, b))


def test_eating_feels_good_starving_feels_bad() -> None:
    # Integration: body → homeostat → affect. A hungry organism at a food patch.
    w = sb.randomize_world(jax.random.key(1), CFG)
    w = w.replace(food_pos=w.food_pos.at[0].set(w.agent_pos), hazard_pos=-w.agent_pos)
    hungry = jnp.asarray(CFG.drive_init).at[hs.ENERGY].set(0.4)
    eat = jnp.zeros(CFG.act_dim).at[sb.MANIPULATE].set(1.0)
    rest = jnp.zeros(CFG.act_dim)

    def mean_energy_valence(action):
        def body(c, k):
            world, h, e = c
            world = sb.step_world(world, action, k, CFG)
            h2 = hs.step_homeostat(h, action, world, jnp.zeros(()), CFG)
            # Only the energy drive differs between the two runs; isolate it.
            d = h.drives.at[hs.ENERGY].set(h2.drives[hs.ENERGY])
            v, _, e = affect.step_affect(d, h.drives, e, G, CFG.dt)
            return (world, h.replace(drives=d, t=h2.t, alive=h2.alive), e), v

        h0 = hs.init_homeostat(CFG).replace(drives=hungry)
        _, v = jax.lax.scan(body, (w, h0, affect.error_magnitude(hungry, G)), jax.random.split(jax.random.key(2), 500))
        return float(v.mean())

    assert mean_energy_valence(eat) > 0
    assert mean_energy_valence(rest) < 0


def test_vmappable() -> None:
    b = 1024
    d = jax.random.uniform(jax.random.key(3), (b, CFG.n_drives))
    v, a, e = jax.vmap(lambda d: affect.step_affect(d, d, affect.error_magnitude(d, G), G, CFG.dt))(d)
    assert v.shape == a.shape == e.shape == (b,)
