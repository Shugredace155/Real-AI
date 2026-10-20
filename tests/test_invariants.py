"""The ten non-negotiable invariants (SPEC Section 2).

This is the highest-priority file in the repo (SPEC 10). Each invariant starts
as a stub that *fails* (not errors) and is filled in as the modules it depends
on are implemented. A stub fails with `pytest.fail`, never `xfail` or `skip`,
so an unimplemented invariant can never read as green.
"""

import pytest


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


def test_rng_explicit() -> None:
    """I9: all randomness flows through explicit JAX PRNG keys; no global RNG."""
    _pending("I9", "the first module that consumes randomness (body/sandbox.py, SPEC 7.3)")


def test_vmappable() -> None:
    """I10: every module is pure-functional and vmaps over a batch dimension."""
    _pending("I10", "the first stepped module (body/sandbox.py, SPEC 7.3)")
