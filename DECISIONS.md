# DECISIONS

Every ambiguity resolved and every addition not in SPEC.md, in the order made.
Format: what was ambiguous, what was chosen, why.

## Step 1 — scaffold, config, CI, invariant stubs

### D1. Toolchain and pinned versions
- **Ambiguity:** SPEC names JAX + Flax but no versions or Python version.
- **Choice:** Python ≥ 3.12 (CI runs 3.13); `jax[cpu]==0.12.0`, `flax==0.12.10`, `pytest==9.1.1`, pinned exactly in `pyproject.toml`.
- **Why:** Exact pins make `run_all_tests(seed=0)` reproducibility (SPEC 10) possible. CPU wheels match "CPU dev"; the GPU extra is added when training starts.

### D2. pytest is the test runner (added dependency)
- **Ambiguity:** SPEC requires automated tests but names no runner.
- **Choice:** pytest, as a `test` extra only (not a runtime dependency).
- **Why:** SPEC test names (`test_*`) already follow pytest's conventions.

### D3. Invariant stubs fail with `pytest.fail`, not `xfail`/`skip`
- **Ambiguity:** SPEC 14 says stubs "should fail, not error".
- **Choice:** Each stub calls `pytest.fail(..., pytrace=False)` naming the invariant and the module it waits on.
- **Why:** `xfail`/`skip` would let CI read green while invariants are unenforced. CI is therefore red until every invariant is implemented — by design, per SPEC 10.

### D4. I8 test name
- **Ambiguity:** SPEC Section 2 gives I8 no test name ("monitored via Growth Ratio"), yet SPEC 14 asks for ten invariant tests.
- **Choice:** `test_genome_no_learnable_competence`, to be implemented as an assertion on the Growth Ratio once fitness and telemetry exist.

### D5. Extra files not in the SPEC 6 layout
- `pyproject.toml` (package metadata + pins), `.github/workflows/ci.yml` (SPEC 10 CI), `.gitignore` (ignores `runs/`, where the genome and life state are written by SPEC 7.15), and `tests/test_config.py` (SPEC 7.1 acceptance needs a home; SPEC 6 lists no config test file). Further per-module test files (e.g. `tests/test_sandbox.py`) will be added as modules land, since SPEC 6 does not list test files for every module.

### D6. `Config.layer_sizes` type
- **Ambiguity:** SPEC types it as bare `tuple`.
- **Choice:** `tuple[int, ...]`, satisfying "type hints everywhere" (SPEC 12). Value unchanged.

### D7. Config field comments are verified by a test
- **Ambiguity:** "all fields documented with a one-line comment" has no stated check.
- **Choice:** `test_all_fields_documented` tokenizes the `Config` source and asserts every dataclass field's line carries a comment.
