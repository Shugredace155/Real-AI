"""Acceptance tests for creature/handoff.py (SPEC 7.15) and its invariants I1, I4."""

import dataclasses
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from _small import SMALL

from creature import handoff as H
from creature.outer import genome as G

CFG = dataclasses.replace(SMALL, log_period=500, checkpoint_period=1_000, growth_window=1_000)
KEY = jax.random.key(CFG.birth_seed)
ROOT = Path(__file__).resolve().parent.parent


def _genome(tmp_path: Path) -> Path:
    path = tmp_path / H.GENOME_FILE
    G.init_genome(jax.random.key(0), CFG).save(path)
    return path


def _same(a, b) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def test_continuity(tmp_path) -> None:
    """I4: kill and restore at step N; the life from N onward is bit-identical."""
    genome = _genome(tmp_path)
    straight = H.run(genome, tmp_path / "straight", CFG, max_steps=3_000, life_key=KEY)
    H.run(genome, tmp_path / "crashed", CFG, max_steps=1_500, life_key=KEY)  # the process dies here
    resumed = H.run(genome, tmp_path / "crashed", CFG, max_steps=3_000, life_key=KEY)  # and is restarted
    assert int(resumed.org.t) == 3_000 and int(resumed.org.sleep_count) == 1
    assert _same(straight, resumed)
    a = H.tm.TelemetryStore(tmp_path / "straight" / "telemetry").read()
    b = H.tm.TelemetryStore(tmp_path / "crashed" / "telemetry").read()
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def test_no_pretrained(tmp_path, monkeypatch) -> None:
    """I1: the only weights read from disk are the genome and this life's own checkpoint."""
    genome = _genome(tmp_path)
    loaded = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda f, *a, **k: (loaded.append(Path(f).name), real_load(f, *a, **k))[1])
    lv = H.run(genome, tmp_path / "life", CFG, max_steps=500, life_key=KEY)
    assert loaded == [H.GENOME_FILE]
    loaded.clear()
    H.run(genome, tmp_path / "life", CFG, max_steps=1_000, life_key=KEY)
    assert set(loaded) == {H.GENOME_FILE, H.LIFE_FILE}
    # Birth weights are exactly the expressed genome.
    born = H.birth(G.Genome.load(genome, CFG), KEY, CFG)
    assert _same(born.org.pcn, G.express(G.Genome.load(genome, CFG), CFG).pcn_init)
    assert int(lv.org.t) == 500


def test_handoff_no_evolution(tmp_path) -> None:
    """Phase 2 never reaches the evolutionary machinery: fitness and evolve are never even imported."""
    genome = _genome(tmp_path)
    script = f"""
import sys, dataclasses
sys.path.insert(0, {str(ROOT / 'tests')!r})
import jax
from _small import SMALL
from creature import handoff as H
cfg = dataclasses.replace(SMALL, log_period=500, checkpoint_period=1_000)
H.run({str(genome)!r}, {str(tmp_path / 'life')!r}, cfg, max_steps=500, life_key=jax.random.key(0))
leaked = [m for m in sys.modules if m.startswith('creature.outer') and m != 'creature.outer.genome' and m != 'creature.outer']
assert not leaked, leaked
print('ok')
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_telemetry_written(tmp_path) -> None:
    lv = H.run(_genome(tmp_path), tmp_path / "life", CFG, max_steps=2_500, life_key=KEY)
    t = H.tm.TelemetryStore(tmp_path / "life" / "telemetry").read()
    assert np.array_equal(t["t"], np.arange(1, 2_501))
    assert np.isfinite(t["growth_ratio"][-1]) and -1 <= t["growth_ratio"][-1] <= 1
    assert int(lv.org.sleep_count) == 1
