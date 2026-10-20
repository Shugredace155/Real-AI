"""The ten non-negotiable invariants (SPEC Section 2).

This is the highest-priority file in the repo (SPEC 10). Each invariant starts
as a stub that *fails* (not errors) and is filled in as the modules it depends
on are implemented. A stub fails with `pytest.fail`, never `xfail` or `skip`,
so an unimplemented invariant can never read as green.
"""

import ast
import importlib
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from creature.config import Config

PACKAGE = Path(__file__).resolve().parent.parent / "creature"
CFG = Config()


def _pending(invariant: str, needs: str) -> None:
    pytest.fail(f"{invariant} not yet implemented: needs {needs}", pytrace=False)


def test_no_pretrained() -> None:
    """I1: no weight tensor is loaded from disk in Phase 2 except the genome."""
    _pending("I1", "handoff.py (SPEC 7.15)")


def test_local_only_awake() -> None:
    """I2: no gradient call anywhere in the awake step; backprop only in sleep.py."""
    _pending("I2", "the full awake step (SPEC 5.1) and memory/sleep.py (SPEC 7.9)")


def test_no_eval_mode() -> None:
    """I3: no train/test split; learning is continuous from timestep 0."""
    _pending("I3", "the full awake step (SPEC 5.1)")


def test_continuity() -> None:
    """I4: checkpoint/restore at step N produces a bit-identical continuation."""
    _pending("I4", "handoff.py state serialization (SPEC 7.15)")


def test_neuromod_causal() -> None:
    """I5: perturbing each of the six neuromodulator channels changes behavior."""
    _pending("I5", "substrate/neuromod.py and agent/policy.py (SPEC 7.6, 7.10)")


def test_valence_is_derivative() -> None:
    """I6: valence is the time derivative of homeostatic error, never its level."""
    _pending("I6", "affect/core.py (SPEC 7.7)")


def test_affect_from_body() -> None:
    """I7: affect is a pure function of body state, with no label/string/prompt input."""
    _pending("I7", "affect/core.py (SPEC 7.7)")


def test_genome_no_learnable_competence() -> None:
    """I8: the genome does not encode learnable competence; monitored via Growth Ratio."""
    _pending("I8", "outer/fitness.py and telemetry.py Growth Ratio (SPEC 7.13, 7.16, 9.3)")


# Files allowed to mint a root PRNG key from a seed. Everything else must receive keys.
KEY_MINTERS = {"handoff.py"}


def _rng_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(PACKAGE).as_posix()
    found = []
    for node in ast.walk(tree):
        where = f"{rel}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "random" or alias.name.startswith("numpy.random"):
                    found.append(f"{where} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "random" or mod.startswith("numpy.random") or (
                mod == "numpy" and any(a.name == "random" for a in node.names)
            ):
                found.append(f"{where} imports from {mod}")
        elif isinstance(node, ast.Attribute):
            if node.attr == "random" and isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"}:
                found.append(f"{where} uses {node.value.id}.random")
            if node.attr in {"key", "PRNGKey"} and rel not in KEY_MINTERS:
                base = node.value
                if isinstance(base, ast.Attribute) and base.attr == "random":
                    found.append(f"{where} mints a PRNG key with random.{node.attr}")
    return found


def test_rng_explicit() -> None:
    """I9: all randomness flows through explicit JAX PRNG keys; no global RNG.

    Static check over the whole package: no Python `random`, no `numpy.random`,
    and no module mints its own key from a seed (keys are passed in), except the
    single Phase 2 entry point. Determinism given a key is checked per module
    (e.g. tests/test_sandbox.py::test_determinism).
    """
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, "package not found"
    violations = [v for f in files for v in _rng_violations(f)]
    assert not violations, "\n".join(violations)


VMAP_BATCH = 1024


def _vmap_sandbox() -> None:
    from creature.body import sandbox as sb

    w = jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(jax.random.key(0), VMAP_BATCH))
    keys = jax.random.split(jax.random.key(1), VMAP_BATCH)
    w = jax.vmap(lambda w, k: sb.step_world(w, jnp.zeros(CFG.act_dim), k, CFG))(w, keys)
    obs = jax.vmap(lambda w: sb.sense(w, jnp.zeros(CFG.n_drives), CFG))(w)
    assert obs.shape == (VMAP_BATCH, sb.obs_dim(CFG))


def _vmap_homeostat() -> None:
    from creature.body import homeostat as hs
    from creature.body import sandbox as sb

    w = jax.vmap(lambda k: sb.randomize_world(k, CFG))(jax.random.split(jax.random.key(0), VMAP_BATCH))
    h = jax.vmap(lambda _: hs.init_homeostat(CFG))(jnp.arange(VMAP_BATCH))
    a = jnp.zeros((VMAP_BATCH, CFG.act_dim))
    h = jax.vmap(lambda h, a, w: hs.step_homeostat(h, a, w, jnp.zeros(()), CFG))(h, a, w)
    assert h.drives.shape == (VMAP_BATCH, CFG.n_drives)


# Every SPEC 7 module with per-organism state or computation. None = not yet
# implemented, which fails. evolve.py and handoff.py are drivers that vmap
# these modules rather than being vmapped themselves (DECISIONS D13).
VMAP_CHECKS: dict[str, Callable[[], None] | None] = {
    "body.sandbox": _vmap_sandbox,
    "body.homeostat": _vmap_homeostat,
    "substrate.pcn": None,
    "substrate.plasticity": None,
    "substrate.neuromod": None,
    "affect.core": None,
    "agent.policy": None,
    "agent.selfmodel": None,
    "memory.episodic": None,
    "memory.sleep": None,
    "outer.genome": None,
    "outer.fitness": None,
    "telemetry": None,
}


@pytest.mark.parametrize("module", list(VMAP_CHECKS))
def test_vmappable(module: str) -> None:
    """I10: every module is pure-functional and vmaps over a batch dimension."""
    check = VMAP_CHECKS[module]
    if check is None:
        _pending("I10", f"creature/{module.replace('.', '/')}.py")
    importlib.import_module(f"creature.{module}")
    check()
