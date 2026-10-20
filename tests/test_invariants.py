"""The ten non-negotiable invariants (SPEC Section 2).

This is the highest-priority file in the repo (SPEC 10). Each invariant starts
as a stub that *fails* (not errors) and is filled in as the modules it depends
on are implemented. A stub fails with `pytest.fail`, never `xfail` or `skip`,
so an unimplemented invariant can never read as green.
"""

import ast
import dataclasses
import importlib
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from _causal import check_channel_causal

from creature.config import Config
from creature.substrate.neuromod import CHANNELS

PACKAGE = Path(__file__).resolve().parent.parent / "creature"
CFG = Config()


def _pending(invariant: str, needs: str) -> None:
    pytest.fail(f"{invariant} not yet implemented: needs {needs}", pytrace=False)


def test_no_pretrained() -> None:
    """I1: no weight tensor is loaded from disk in Phase 2 except the genome."""
    _pending("I1", "handoff.py (SPEC 7.15)")


GRADIENT_APIS = {"grad", "value_and_grad", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linear_transpose"}


def test_local_only_awake() -> None:
    """I2: no gradient call anywhere in the awake step; backprop only in sleep.py.

    Every gradient API in the package appears only in memory/sleep.py, and the
    waking step (organism.step) never references sleep.
    """
    users = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        if names & GRADIENT_APIS:
            users.append(path.relative_to(PACKAGE).as_posix())
    assert users == ["memory/sleep.py"], users
    organism = ast.parse((PACKAGE / "organism.py").read_text())
    step = next(n for n in organism.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    referenced = {n.id for n in ast.walk(step) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(step) if isinstance(n, ast.Attribute)
    }
    assert not {"sleep", "sl", "consolidate_bptt"} & referenced
    imported = {(n.module or "") for n in organism.body if isinstance(n, ast.ImportFrom)}
    assert "creature.memory.sleep" not in imported


MODE_FLAGS = {"train", "training", "is_training", "eval", "evaluate", "eval_mode", "deterministic", "inference"}


def test_no_eval_mode() -> None:
    """I3: no train/test split; learning is continuous from timestep 0.

    No function in the package takes a mode flag, and the newborn's very first
    waking step already changes its weights.
    """
    from _small import SMALL

    from creature import organism as O
    from creature.body import sandbox as sb

    for path in sorted(PACKAGE.rglob("*.py")):
        for fn in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)):
            args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            assert not args & MODE_FLAGS, f"{path.name}:{fn.name} takes a mode flag {args & MODE_FLAGS}"
    g = O.default_endowment(jax.random.key(0), SMALL)
    org, _, _ = O.step(O.spawn(g, SMALL), sb.randomize_world(jax.random.key(1), SMALL), jax.random.key(2), SMALL)
    assert any(float(jnp.abs(a - b).max()) > 0 for a, b in zip(jax.tree.leaves(org.pcn), jax.tree.leaves(g.pcn_init)))


def test_continuity() -> None:
    """I4: checkpoint/restore at step N produces a bit-identical continuation."""
    _pending("I4", "handoff.py state serialization (SPEC 7.15)")


@pytest.mark.parametrize("channel", range(CFG.n_neuromod), ids=CHANNELS)
def test_neuromod_causal(channel: int) -> None:
    """I5: clamping each neuromodulator channel to 0 vs max changes a computational parameter.

    Probes live in tests/_causal.py; a channel whose knob module is not built yet fails.
    """
    check_channel_causal(channel)


def test_valence_is_derivative() -> None:
    """I6: valence is the time derivative of homeostatic error, never its level."""
    from _birth import birth_genome

    from creature.affect import core as affect

    g = birth_genome()
    dt = CFG.dt
    feel = lambda d, prev: affect.step_affect(d, prev, affect.error_magnitude(prev, g), g, dt)[0]
    below = jnp.clip(g.setpoint - 0.3, 0.0, 1.0)  # every drive below its setpoint
    nudge = jnp.zeros(CFG.n_drives).at[0].set(1e-3)  # energy moves toward its setpoint

    # Same level now, different history -> different valence.
    assert float(feel(below, below - nudge)) > 0 > float(feel(below, below + nudge))
    # Different levels, same change -> same valence.
    v_far, v_near = feel(below + nudge, below), feel(below + 0.1 + nudge, below + 0.1)
    assert float(v_far) == pytest.approx(float(v_near), rel=1e-2)
    # Any constant body, comfortable or not, feels exactly nothing.
    for d in jax.random.uniform(jax.random.key(0), (100, CFG.n_drives)):
        assert float(feel(d, d)) == 0.0


def test_affect_from_body() -> None:
    """I7: affect is a pure function of body state, with no label/string/prompt input.

    Valence and arousal take only drive vectors and evolved setpoints. The
    neuromodulator drivers take only affect outputs, the organism's own
    prediction error and caregiver proximity (a body-contact signal).
    """
    import inspect

    from creature.affect import core as affect
    from creature.substrate import neuromod as nmod

    assert list(inspect.signature(affect.step_affect).parameters) == ["drives", "prev_drives", "prev_err_mag", "g", "dt"]
    assert list(inspect.signature(nmod.neuromod_target).parameters) == [
        "valence", "arousal", "pred_err", "social_presence", "g"
    ]
    for path in (PACKAGE / "affect" / "core.py", PACKAGE / "substrate" / "neuromod.py"):
        tree = ast.parse(path.read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert mods <= {"jax", "flax", "typing", "__future__"}, f"{path.name} imports {mods}"
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for arg in fn.args.args:
                assert not (arg.annotation and "str" in ast.unparse(arg.annotation)), f"{fn.name}({arg.arg}) takes text"


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


def _vmap_pcn() -> None:
    from creature.substrate import pcn

    sizes = pcn.pcn_sizes(16, CFG)
    params, state = jax.vmap(lambda k: pcn.init_pcn(k, sizes, CFG))(jax.random.split(jax.random.key(0), VMAP_BATCH))
    one = jnp.array(1.0)
    out, state = jax.vmap(lambda p, s: pcn.pcn_forward(p, s, jnp.ones(sizes[0]), one, CFG))(params, state)
    params = jax.vmap(lambda p, s, e: pcn.pcn_update(p, s, e, one, one, CFG))(params, state, out.errors)
    assert out.pred_err.shape == (VMAP_BATCH,)
    assert params.W_down[0].shape == (VMAP_BATCH, sizes[0], sizes[1])


def _vmap_plasticity() -> None:
    from _birth import birth_genome

    from creature.substrate import plasticity as pl

    g = birth_genome()
    w = jnp.zeros((VMAP_BATCH, 4, 3))
    gate = jnp.ones(VMAP_BATCH)
    pre, post = jnp.ones((VMAP_BATCH, 3)), jnp.ones((VMAP_BATCH, 4))
    w, meta = jax.vmap(lambda w, p, q, gt: pl.plastic_update(w, w, w, p, q, gt, g, CFG.dt))(w, pre, post, gate)
    assert w.shape == meta.shape == (VMAP_BATCH, 4, 3)


def _vmap_neuromod() -> None:
    from _birth import birth_genome

    from creature.substrate import neuromod as nmod

    g = birth_genome()
    x = jnp.zeros(VMAP_BATCH)
    st = jax.vmap(lambda v: nmod.step_neuromod(nmod.init_neuromod(g), v, v, v, v, g, CFG.dt))(x)
    assert st.nm.shape == (VMAP_BATCH, CFG.n_neuromod)
    assert jax.vmap(nmod.plasticity_gate)(st.nm).shape == (VMAP_BATCH,)


def _vmap_affect() -> None:
    from _birth import birth_genome

    from creature.affect import core as affect

    g = birth_genome()
    d = jnp.full((VMAP_BATCH, CFG.n_drives), 0.5)
    e = jax.vmap(lambda d: affect.error_magnitude(d, g))(d)
    v, a, e = jax.vmap(lambda d, p, e: affect.step_affect(d, p, e, g, CFG.dt))(d, d, e)
    assert v.shape == a.shape == e.shape == (VMAP_BATCH,)


def _vmap_policy() -> None:
    from creature import organism as O
    from creature.agent import policy as pol
    from creature.substrate import neuromod as nmod

    g = O.default_endowment(jax.random.key(0), CFG)
    z = jnp.zeros((VMAP_BATCH, pol.z_dim(CFG)))
    keys = jax.random.split(jax.random.key(1), VMAP_BATCH)
    kn = nmod.knobs(jnp.ones(CFG.n_neuromod))
    e = jnp.zeros(CFG.n_drives)
    a, logits = jax.vmap(lambda z, k: pol.policy(pol.init_dyn(CFG), z, e, kn, g, CFG, k))(z, keys)
    assert a.shape == (VMAP_BATCH,) and logits.shape == (VMAP_BATCH, pol.n_actions(CFG))


def _vmap_selfmodel() -> None:
    from creature.agent import selfmodel as sm

    p = sm.init_selfmodel(CFG)
    s = jnp.zeros((VMAP_BATCH, sm.self_input_dim(CFG)))
    err = jax.vmap(lambda s: sm.self_model(p, s, jnp.zeros(CFG.act_dim), jnp.zeros(sm.self_state_dim(CFG))))(s)
    assert err.shape == (VMAP_BATCH,)


def _vmap_episodic() -> None:
    from creature.memory.episodic import Episodic

    ep = jax.vmap(lambda _: Episodic.create(16, 3, 5))(jnp.arange(VMAP_BATCH))
    ep = jax.vmap(lambda e, x: e.write(jnp.full(3, x), jnp.full(5, x), x, x).tick(CFG.dt))(ep, jnp.arange(VMAP_BATCH) * 1.0)
    k, v, _ = jax.vmap(lambda e, r: e.replay_batch(r, 4, CFG))(ep, jax.random.split(jax.random.key(0), VMAP_BATCH))
    assert k.shape == (VMAP_BATCH, 4, 3) and v.shape == (VMAP_BATCH, 4, 5)


def _vmap_sleep() -> None:
    from _small import SMALL

    from creature import organism as O
    from creature.memory import sleep as sl
    from creature.memory.episodic import Episodic

    b = 8  # a night of consolidation per organism is heavy; batch kept small
    gs = jax.vmap(lambda k: O.default_endowment(k, SMALL))(jax.random.split(jax.random.key(0), b))
    dk, dv = O.latent_dim(SMALL), SMALL.episodic_seq_len * (O.sb.obs_dim(SMALL))
    ep = jax.vmap(lambda _: Episodic.create(4, dk, dv).write(jnp.zeros(dk), jnp.ones(dv), 1.0, 1.0))(jnp.arange(b))
    cfg = dataclasses.replace(SMALL, sleep_consolidation_steps=2, sleep_replay_batch=2)
    out = jax.vmap(lambda p, e, k: sl.sleep(p, e, k, cfg))(gs.pcn_init, ep, jax.random.split(jax.random.key(1), b))
    assert jax.tree.map(jnp.shape, out) == jax.tree.map(jnp.shape, gs.pcn_init)


# Every SPEC 7 module with per-organism state or computation. None = not yet
# implemented, which fails. evolve.py and handoff.py are drivers that vmap
# these modules rather than being vmapped themselves (DECISIONS D13).
VMAP_CHECKS: dict[str, Callable[[], None] | None] = {
    "body.sandbox": _vmap_sandbox,
    "body.homeostat": _vmap_homeostat,
    "substrate.pcn": _vmap_pcn,
    "substrate.plasticity": _vmap_plasticity,
    "substrate.neuromod": _vmap_neuromod,
    "affect.core": _vmap_affect,
    "agent.policy": _vmap_policy,
    "agent.selfmodel": _vmap_selfmodel,
    "memory.episodic": _vmap_episodic,
    "memory.sleep": _vmap_sleep,
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
