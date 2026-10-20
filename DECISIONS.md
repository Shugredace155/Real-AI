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

## Step 2 — body/sandbox.py, body/homeostat.py

### D8. Actions are a continuous 4-vector; metabolic cost interpolates the SPEC table
- **Ambiguity:** SPEC 7.2 indexes `METABOLIC_COST[action]` by discrete names (rest/locomote/manipulate/vocalize), while SPEC 7.10 builds continuous `(N_ACTIONS, ACT_DIM)` candidates. A string-keyed dict is also not jittable.
- **Choice:** `action ∈ [-1, 1]^4 = [thrust, turn, manipulate, vocalize]`. Locomote intensity is `max(|thrust|, |turn|)`; manipulate/vocalize intensities are the positive part of their component. Cost/s = `rest + Σ (cost_e − rest) · intensity_e`, so a pure single-effector action at full intensity reproduces the table exactly (`test_metabolic_cost_matches_table`). The table values are `Config.metabolic_cost_*`.

### D9. `step_homeostat(state, action, world, pred_err, cfg)` signature
- **Ambiguity:** SPEC shows `step_homeostat(state, action, env, g, dt)` and reads `state.recent_pred_err`, but nothing writes that buffer; `g` is used elsewhere in the SPEC for both genome and config.
- **Choice:** Pass the step's total prediction error explicitly; the homeostat pushes it into a `pred_err_window`-step ring buffer that feeds novelty (mean → information intake) and safety (variance). `cfg` replaces `g`/`dt` (dt is `cfg.dt`). No genome input: setpoints/urgency are applied in affect (SPEC 7.7). `ingest`, `damage`, `ambient` and `proximity` are computed by `step_world` and read from the post-step world.

### D10. Sandbox contents and the scripted caregiver
- **Ambiguity:** SPEC 7.3 lists "a few movable objects, one caregiver, one hazard" but energy "refills on ingestion" with no food source named.
- **Choice:** Added `n_food = 3` regrowing food patches as the only energy source; ingestion = manipulating the nearest stocked patch within reach. Movable objects can be pushed or pulled by manipulating. The caregiver is scripted (SPEC 13 default): it keeps a randomized preferred distance, approaches faster when the organism vocalizes, wanders with an Ornstein–Uhlenbeck velocity, and warms the ambient temperature near itself. It is the "nearest conspecific" for the social drive. Collision model is deliberately simple: inelastic organism–object contact, positional-only object–object separation, kinematic caregiver. Spawn positions may overlap (the organism can be born touching the hazard).

### D11. SPEC metabolic numbers kept as-is — flagged for Phase 1e
- At 0.1 energy/s resting cost, a full organism starves in 10 s (10^4 steps); `dev_steps` is 200 s. This is harsh but is what SPEC 7.2 specifies. Revisit in Phase 1e if survival dominates fitness.

### D12. Throughput is measured as total env-steps/sec across the batch
- **Ambiguity:** "≥ 20,000 env steps/sec on CPU for batch 4096".
- **Choice:** `batch × steps / seconds`, including `sense` (the realistic per-step cost), measured after compilation. SPEC 12's GPU figure (500k) only makes sense under this reading. Measured ≈ 630k–770k on the 4-core dev container.

### D13. I10 is a per-module registry
- `test_vmappable` is parametrized over every SPEC 7 module with per-organism computation; a module with no check yet fails. `outer/evolve.py` and `handoff.py` are excluded: they are drivers that vmap other modules rather than being vmapped.

### D14. I9 is a static scan of the package
- `test_rng_explicit` parses every file under `creature/` and fails on Python `random`, `numpy.random`, or minting a key with `jax.random.key/PRNGKey`. Only `handoff.py` (which owns `BIRTH_SEED`) may mint a root key. Determinism given a key is tested per module.

### D15. Drive details
- `temperature` is body temperature, not satisfaction; its goodness is distance to the evolved setpoint (SPEC 7.7).
- Safety target is `clip(1 − safety_var_gain · var(window), 0, 1)` with gain 4, because the variance of values in [0, 1] is at most 0.25 and safety must be losable.
- A dead body's drives are frozen and `alive` is sticky.
- `test_every_drive_losable` (added) shows each drive can reach 0.

### D16. The boredom test uses a stand-in predictor until the PCN exists
- **Ambiguity:** SPEC 7.2's boredom needs prediction error, which comes from SPEC 7.4 (not yet built).
- **Choice:** `test_boredom` uses an exponential running average of the observation as the predictor, in a world with a frozen caregiver and a resting organism, and refills energy/integrity so death does not freeze novelty first. Novelty falls from 0.5 to < 0.1 at ≈ 16k steps. Re-run with the real PCN in step 3.

### D17. Extra test files and tests
- `tests/test_sandbox.py`, `tests/test_homeostat.py` (SPEC 6 lists neither). Added beyond the acceptance list: `test_observation_is_low_level`, `test_bodies_stay_in_arena`, `test_ingestion_refills_energy`, `test_metabolic_cost_matches_table`, `test_every_drive_losable`, `test_dead_stays_dead`.
