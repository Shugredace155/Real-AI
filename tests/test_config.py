"""Acceptance tests for creature/config.py (SPEC 7.1)."""

import dataclasses
import inspect
import io
import tokenize

import pytest

from creature import config
from creature.config import Config


def _field_comment_lines() -> dict[str, bool]:
    """Map each annotated field name in Config's source to whether its line has a comment."""
    src = inspect.getsource(Config)
    has_comment: dict[int, bool] = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            has_comment[tok.start[0]] = True
    result = {}
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        name = stripped.split(":", 1)[0]
        if ":" in stripped and name.isidentifier():
            result[name] = has_comment.get(lineno, False)
    return result


def test_config_constructs() -> None:
    cfg = Config()
    assert dataclasses.is_dataclass(cfg)


def test_config_frozen() -> None:
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dt = 0.5  # type: ignore[misc]


def test_config_hashable() -> None:
    # Hashability is what lets Config be a static argument to jax.jit.
    assert hash(Config()) == hash(Config())


def test_all_fields_documented() -> None:
    comments = _field_comment_lines()
    for f in dataclasses.fields(Config):
        assert comments.get(f.name), f"Config.{f.name} lacks a one-line comment"


def test_spec_defaults() -> None:
    # Pin the SPEC 7.1 defaults so drift is a deliberate, logged decision.
    cfg = Config()
    assert cfg.dt == 1e-3
    assert cfg.n_drives == 6
    assert cfg.n_neuromod == 6
    assert cfg.layer_sizes == (64, 128, 128, 64)
    assert cfg.n_fast_synapses == 0
    assert cfg.episodic_capacity == 100_000
    assert cfg.sleep_period == 60_000
    assert cfg.sleep_replay_batch == 512
    assert cfg.sleep_consolidation_steps == 200
    assert cfg.pop_size == 512
    assert cfg.n_eval_worlds == 8
    assert cfg.dev_steps == 200_000
    assert cfg.generations == 4000
    assert (cfg.w_survival, cfg.w_learn, cfg.w_competence) == (1.0, 3.0, 1.0)
    assert (cfg.w_robustness, cfg.w_curiosity, cfg.w_hack) == (2.0, 0.5, 0.5)


def test_module_has_no_global_state() -> None:
    # Only the Config class (and imports) should be public at module level.
    public = {n for n in vars(config) if not n.startswith("_")}
    assert public <= {"Config", "preset", "dataclass", "replace", "annotations"}
