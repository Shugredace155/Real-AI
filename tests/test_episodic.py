"""Acceptance tests for creature/memory/episodic.py (SPEC 7.8)."""

import jax
import jax.numpy as jnp
import numpy as np

from creature.config import Config
from creature.memory.episodic import Episodic, load_episodic, save_episodic

CFG = Config()


def _filled(n: int, valences: jnp.ndarray) -> Episodic:
    ep = Episodic.create(n, 3, 4)
    for i, v in enumerate(valences):
        ep = ep.write(jnp.full(3, i * 1.0), jnp.full(4, -i * 1.0), v, jnp.array(0.0)).tick(CFG.dt)
    return ep


def test_write_once() -> None:
    ep = _filled(8, jnp.zeros(5))
    for i in range(5):
        assert float(ep.k[i, 0]) == i and float(ep.v[i, 0]) == -i
    # Writing more never touches existing slots until the ring wraps.
    ep2 = ep.write(jnp.full(3, 99.0), jnp.full(4, 99.0), 0.0, 0.0)
    assert np.array_equal(np.asarray(ep2.k[:5]), np.asarray(ep.k[:5]))
    assert int(ep2.ptr) == 6 and float(ep2.k[5, 0]) == 99.0


def test_ring_wraps() -> None:
    ep = _filled(4, jnp.zeros(6))
    assert int(ep.fill) == 4 and int(ep.ptr) == 2
    assert float(ep.k[0, 0]) == 4 and float(ep.k[1, 0]) == 5  # oldest slots overwritten first


def test_salience_bias() -> None:
    # Half the traces are emotionally intense (|valence| = 2), half mundane (0.1).
    n = 100
    val = jnp.where(jnp.arange(n) % 2 == 0, 2.0, 0.1) * jnp.where(jnp.arange(n) % 4 == 0, -1.0, 1.0)
    ep = _filled(n, val)
    _, _, idx = ep.replay_batch(jax.random.key(0), 100_000, CFG)
    counts = np.bincount(np.asarray(idx), minlength=n)
    high, low = counts[::2].sum(), counts[1::2].sum()
    print(f"high-|valence| samples {high}, low {low} ({high / low:.1f}x)")
    assert high >= 3 * low


def test_never_replays_unwritten() -> None:
    ep = _filled(64, jnp.ones(10))
    _, _, idx = ep.replay_batch(jax.random.key(1), 10_000, CFG)
    assert int(idx.max()) < 10


def test_decay() -> None:
    ep = _filled(4, jnp.ones(4))
    history = []
    for _ in range(50):
        ep = ep.tick(1.0)
        history.append(np.asarray(ep.effective_salience(CFG)))
    h = np.stack(history)
    assert np.all(np.diff(h, axis=0) < 0), "untouched traces must lose salience monotonically"


def test_serialize_roundtrip(tmp_path) -> None:
    ep = _filled(16, jax.random.normal(jax.random.key(2), (20,)))
    path = tmp_path / "episodic.npz"
    save_episodic(ep, path)
    back = load_episodic(path)
    for f in ep.__dataclass_fields__:
        a, b = np.asarray(getattr(ep, f)), np.asarray(getattr(back, f))
        assert a.dtype == b.dtype and np.array_equal(a, b), f
