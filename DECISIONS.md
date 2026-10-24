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

## Step 3 — substrate/pcn.py, substrate/plasticity.py, substrate/neuromod.py

### D18. PCN layers: the observation is layer 0, `layer_sizes` are the latent layers
- **Ambiguity:** `layer_sizes = (64, 128, 128, 64)` but the observation is 77-dimensional.
- **Choice:** Layer widths are `(obs_dim,) + layer_sizes` = (77, 64, 128, 128, 64). Layer 0 is clamped to the observation; the top layer (64) is a free latent with a Gaussian prior (`pcn_top_prior`).

### D19. `W_up` is a learned feedback matrix (Kolen–Pollack); not inherited
- **Ambiguity:** SPEC 7.4 says "two weight matrices" per layer but its code uses three (`W_down`, `W_lat`, `W_up`), and `W_up[l] += outer(act[l], errors[l])` has an inconsistent index.
- **Choice:** `W_up[l]` (n_{l+1}, n_l) carries errors bottom-up during relaxation instead of reading `W_down[l]ᵀ` across layers. It receives exactly the transpose of `W_down`'s update (`outer(φ(act[l+1]), err[l])`, i.e. the SPEC's line with the index corrected), and both decay at `pcn_weight_decay`, so any mismatch shrinks exponentially (`test_w_up_tracks_transpose`). It is born as `W_down[l]ᵀ` and is not a separate gene: with it, the genome would be ≈ 356k > 300k.
- **Implication for sleep (step 6):** sleep updates `W_down`; the shared decay re-aligns `W_up` afterwards, at rate `lr · pcn_weight_decay` (≈ 0.027/s at birth). Revisit if that is too slow.

### D20. One PCN timestep: top-down sweep, then K relaxation iterations
- Each timestep the latent layers start at their prior prediction (top-down sweep from the top latent, which persists between steps, plus lateral context), then relax for `pcn_inference_steps = 5` iterations. This is the Whittington & Bogacz scheme and makes errors exactly 0 above the sensory layer before relaxation.
- `pred_err` (fed to affect and the drives) is the mean squared *prior* sensory error, the surprise before the observation is explained away. The returned `errors` are post-relaxation and are the learning signal.
- `precision` is an explicit argument (ACh-gated by the policy knobs in step 5), applied to errors in both inference and learning.
- Biases were added (the SPEC's sketch has none); without them a constant input cannot be predicted.

### D21. Two update entry points
- `pcn_deltas` returns per-second local learning signals shaped like the params. The awake loop routes these through `plasticity.apply_delta` (gate + metaplasticity). `pcn_update` is the plain, ungated `params + dt·lr·deltas` and is used to test the learning algorithm itself.

### D22. Oja term is row-wise
- **Ambiguity:** SPEC 7.5's `w * jnp.outer(post, post)` only broadcasts when `w` is square.
- **Choice:** Oja's rule as published: `outer(post, pre) − post²[:, None] · w`.

### D23. Metaplasticity hardens on *expressed* plasticity (deviation)
- **SPEC:** `meta += dt·(|hebb|/meta_tau − meta·META_DECAY)`, which hardens even while the gate is closed.
- **Choice:** `meta += dt·(|gate·delta|/meta_tau − meta·meta_decay)`. A synapse "used" = a synapse that changed. This makes `test_neuromod_gates` exact (DA = 0 freezes weights *and* metaplastic state; meta only relaxes) and stops a statue genome (gate ≈ 0) from hardening anyway.

### D24. Forgetting resistance: tested with the error-driven rule
- **Finding:** with the Oja rule, metaplasticity cannot give ≥ 70% retention *and* learn task B. Oja normalizes each row, so learning B on shared postsynaptic units *requires* eroding A, and hardening only trades speed for retention (scan: retention ≥ 0.7 only where B reached ≤ 0.3).
- **Choice:** `test_forgetting_resistance` uses `apply_delta` with an error-driven delta (the rule the PCN uses). Task B (`[c, c] → −M c`) directly contradicts task A (`[a, 0] → M a`) on the shared half but has many solutions. Without metaplasticity, A retains 0.00; with the birth defaults it retains 0.71–0.75 across 5 seeds while learning B to ≥ 0.98. The margin over 0.70 is thin; `meta_tau` is evolved per synapse.
- Birth defaults changed to make this hold: `init_log_lr = 1.0` (was 0.0), `init_meta_tau = 0.1` (was 1.0).

### D25. Neuromodulator details
- `step_neuromod` takes `nm_gain` and `nm_tau` explicitly (the SPEC's `g`). Thresholds (`cort_pred_err_threshold = 0.3`, `sht_arousal_ref = 0.5`) and the CORT suppression factor are in `Config`.
- Time constants below `dt` are floored at `dt`, so an evolved `nm_tau` can never make the Euler step overshoot.
- Birth levels `nm_init = (0.5, 0.5, 0.5, 0.5, 0.5, 0)`; birth time constants (0.2, 5, 0.1, 0.5, 10, 20) s span 200×.

### D26. OPEN CONCERN — DA-gated plasticity is ~0 by default (not changed; needs a decision)
- Per SPEC, DA's target is `clip(valence, −1, 1) · gain` and levels are clipped at 0. Valence is a derivative (I6), so at steady state DA → 0, and during worsening (negative valence) DA → 0 as well. The plasticity gate is *linear* in DA, so **the organism learns only while its homeostatic error is actively falling**. In a stable or deteriorating situation it is a statue, which works against `learn_speed` (SPEC 7.13).
- Not changed, because it is a design question rather than an ambiguity. Candidate fixes for step 5 / Phase 1e: (a) a tonic DA baseline `target = nm_base + drive · gain` with an evolved `nm_base`; (b) drive DA with `|valence|` (salience) instead of signed valence; (c) leave as is and let evolution raise `nm_gain`.

### D27. I5 probes
- `tests/_causal.py` maps each channel to the knobs it gates. DA, NE, ACh and CORT are causal now through the plasticity gate. 5HT (discount) and OT (caregiver weight) have no knob until `agent/policy.py`, so their I5 checks fail as pending.

### D28. Boredom now uses the real PCN (supersedes D16)
- `test_boredom` runs the real PCN online in a static world. `info_gain` raised 1 → 10 because an untrained PCN's surprise is ≈ 0.05; at gain 1 novelty fell below 0.1 even with learning off, so the test was vacuous. Added `test_boredom_requires_learning`: with plasticity off, novelty stays above 0.1.

### D29. Extra tests (beyond the acceptance list)
- PCN: `test_relaxation_assigns_credit`, `test_learns_temporal_sequence`, `test_w_up_tracks_transpose`. Plasticity: `test_meta_reopens`, `test_forgetting_needs_metaplasticity`. Neuromod: `test_tracks_driver`, `test_timescales_order_response`. Homeostat: `test_boredom_requires_learning`.

### D30. Projected genome dimension
- With the PCN fixed: P = 80,886 inherited synapses (W_down 37,696 + W_lat 42,793 + biases 397; W_up excluded per D19). `init_w`, `log_lr` and `meta_tau` are each (P,), so 3P = 242,658, plus the small vectors. That leaves ≈ 57k headroom under the 300k bound for any inherited policy/latent-dynamics weights in step 5; if exceeded, SPEC 13 #2 (low-rank `init_w`) applies.

## Step 3 revision — tonic + |valence| DA, floored plasticity gate (author's direction)

### D31. DA is tonic plus |valence|; every gate factor is floored (resolves D26)
- DA is tonic plus |valence|, unsigned. Signed valence is preserved in telemetry and will drive an eligibility trace in Phase 2, since an unsigned gate cannot do delayed credit assignment.
- CORT mildly suppresses plasticity by design, so chronic stress produces reduced learning and learned helplessness becomes an observable failure mode rather than a designed-away one.
- Every factor in the plasticity gate carries an additive floor because a multiplicative gate over neuromodulators is otherwise a chain of vetoes. Minimum gate = 0.10 · 1.0 · 0.30 · 0.30 = 0.009 (`test_gate_never_zero`).
- New genome field `da_tonic` (1,), prior uniform on [0.1, 1.0] (`Config.da_tonic_prior`). `outer/genome.py` (step 7) must include it; until then tests use the stand-in `tests/_birth.py`.
- `knobs(nm)` now lives in `substrate/neuromod.py` (SPEC 7.6 placed the mapping in `agent/policy.py`) and is the I5 check: every channel moves at least one knob by > 1e-3. Once the policy exists, I5 should also probe the action distribution.

### D32. Adaptations made while implementing D31
- **Oja term kept row-wise.** The directed code repeats SPEC's `w * jnp.outer(post, post)`, which only broadcasts for square `w` (D22). Kept `post²[:, None] · w`.
- **`nm_tau` floored at `dt`** (kept from D25). The directed code divides by `g.nm_tau` directly; the floor only matters if evolution produces a time constant below 1 ms, where the Euler step would overshoot.
- **Constants are module-level**, as directed (`DA_FLOOR`, `META_DECAY`, knob offsets, …), departing from SPEC 12's "everything in config.py". Knob offsets/scales were given names rather than inline literals. The superseded Config fields (`meta_decay`, `cort_plasticity_suppression`, `nm_init`, `nm_max`, `sht_arousal_ref`, `cort_pred_err_threshold`) were removed.
- **`init_log_lr` 1.0 → 0.0**: log learning rates are now clipped to [−12, 0], so 1.0 would be clipped to 0 anyway. Birth lr is 1/s.
- **`apply_delta` takes the gate** (computed once per step) like `plastic_update`; the PCN routes its deltas through it the same way.
- **`test_gate_zero_freezes`**: "meta unchanged" is read as "meta never rises". Passive decay (synapses reopening) runs regardless of the gate by design, so a nonzero meta drifts down. The test starts from nonzero meta and asserts weights are bit-identical and meta never increases.

### D33. OPEN — `test_forgetting_resistance` fails under the new criterion
- **Criterion as directed:** task A for 10⁴ steps, then task B for 10⁴ steps; A's error after B must be ≤ 1.3 × its post-A error.
- **Result at birth defaults** (contradicting task from D24, birth gate 0.594): A error 3.236 → 0.885 after A (only 73% learned) → 1.420 after B, a ratio of 1.60. **Fails.**
- **The ratio is vacuous on its own:** a synapse that barely learns A also barely forgets it (ratio → 1). The test therefore also requires that A be ≥ 90% learned; without that guard, raising hardening makes the test pass by making the organism a statue. Where A *is* learned well, post-A error ≈ 0, so any interference at all fails the ratio.
- **META_DECAY is not the lever:** a sweep shows larger decay *reduces* retention (share of A's improvement kept after B, at `meta_tau` 0.3: 0.66 / 0.64 / 0.46 / 0.00 for decay 0.001 / 0.01 / 0.1 / 1.0), because decay is what lets hardened synapses reopen.
- **Learning budget shrank:** lr is capped at 1/s and the birth gate is 0.594, so 10⁴ steps give ≈ 6 lr·s of learning, versus ≈ 54 under D24's setup. That is why A is no longer fully learned in one phase.
- A literal pass exists only in a narrow corner (target noise 0.5, an independent rather than contradicting B, `meta_tau` 0.3: ratio 1.25 with A 93% learned). Choosing that would be tuning the test to the answer, so it was not adopted. Needs the author's decision.
- `test_forgetting_needs_metaplasticity` (hardening forgets less than no hardening) still passes.

### D34. Forgetting resistance measured as retention of improvement (resolves D33)
- **Author's choice:** `retention = (initial − A error after B) / (initial − A error after A) ≥ 0.70`, with 10⁴ steps per task at the birth gate.
- Kept a looser non-vacuity guard: A must be ≥ 50% learned, since a synapse that barely learns also barely forgets.
- Result at birth defaults: A 70–74% learned, retention 0.77–0.78 across 5 seeds (B contradicts A). Without metaplasticity A is wiped out (control test).

## Step 4 — affect/core.py

### D35. Affect follows SPEC 7.7 exactly, plus a birth helper
- `step_affect` is the SPEC code. Added `error_magnitude(drives, g)` so the caller can seed `prev_err_mag` at birth. Otherwise the first valence would be `−err_mag/dt`, a huge spurious spike rather than a derivative.
- Birth setpoints `(0.8, 0.5, 1.0, 0.6, 0.4, 0.8)` and urgencies (all 1) are `Config.setpoint_init` / `urgency_init`; the genome evolves them (step 7). `tests/_birth.py` carries them until then.
- Error is `|drive − setpoint|`, symmetric, so surplus counts as error: at birth energy is 1.0 > setpoint 0.8, so burning energy down toward 0.8 has positive valence. This is what the SPEC formula implies; noted, not changed.

### D36. Concern — the arousal "change" term is negligible (not changed)
- `arousal = tanh(Σ|drives − prev_drives| + err_mag)`. The change term is per timestep, so O(dt) ≈ 10⁻⁴, while `err_mag` is O(1); arousal is effectively `tanh(err_mag)`, a level. The glossary defines arousal as "magnitude of *unpredicted* change plus drive urgency". Candidates for Phase 1e: use the rate `Σ|Δdrives|/dt`, or the organism's own prediction error, as the change term.

### D37. Extra affect tests
- `test_valence_is_rate_per_second` (valence is in units of 1/s, independent of dt) and `test_eating_feels_good_starving_feels_bad` (sandbox → homeostat → affect integration).
- Invariants I6 (`test_valence_is_derivative`) and I7 (`test_affect_from_body`) are now real tests. I7 checks signatures, that affect and neuromod import only jax/flax/typing, and that no function takes a text-typed argument.

## Step 5 — agent/policy.py, agent/selfmodel.py, creature/organism.py

### D38. `metabolic_scale = 0.01` (deviation from SPEC 7.2 rates; Phase 1e lever)
- At SPEC rates (rest 0.1/s … manipulate 0.8/s) a newborn with a full energy store and an exploring policy starved in 2.1 s. No learning can happen in a 2-second life, so the developmental program (SPEC 1, 7.13) is impossible at those rates; D11 had flagged this.
- All metabolic costs are now multiplied by `Config.metabolic_scale = 0.01`, keeping the SPEC table's ratios: resting starvation takes ≈ 17 min, full manipulation ≈ 2 min. `test_metabolic_cost_matches_table` and `test_starvation` check both the SPEC rates (scale 1) and the configured scale.

### D39. New file `creature/organism.py` (not in the SPEC 6 layout)
- Holds the `Endowment` (the inherited phenotype, glossary "Endowment"), the `Organism` state, `spawn`, and `step`, the SPEC 5.1 control loop. `fitness.py` and `handoff.py` both need this loop, and putting it in either would make the other import across the Phase 1 / Phase 2 boundary. `organism.py` never imports `creature.outer`.
- Loop order is exactly SPEC 5.1, with one added step 0: the world advances under the action currently held, before `sense`. The SPEC loop has no world step.
- Per-step keys are `fold_in(life_key, t)`, a pure function of the clock, so a restored life draws identical randomness (for I4).
- A dead organism's gate is 0 and it rests; its brain no longer learns.

### D40. Policy: motor primitives, held for a policy period, scored by a learned forward model
- **Candidates:** SPEC 7.10's `linspace(-1, 1, N)[:, None] * ones((1, ACT_DIM))` sets thrust = turn = manipulate = vocalize, a degenerate one-dimensional repertoire. Replaced by 10 motor primitives (`Config.motor_primitives`: rest, forward, back, turn L/R, forward-turn L/R, grab, approach-grab, vocalize).
- **Temporal abstraction:** a chosen primitive is held for `policy_period = 100` steps (100 ms). At 1 ms per step, a single step's effect on drives is below noise, so a forward model at the raw timestep could not distinguish actions. Holding a motor primitive for 100 ms is not turn-taking: the clock still runs every millisecond (SPEC 3).
- **Forward model** (`predict_rollout` in the SPEC): a single linear layer over [tanh(z), action one-hot, one-hot ⊗ tanh(z[:8])], predicting per-period change of latent, change of the homeostatic error vector, mean caregiver proximity and mean prediction error. It is learned once per completed period by a local delta rule through `apply_delta` (I2), with the gate and metaplasticity. Its initial weights are inherited (genome `init_w`).
- **Latent world state** z = the first latent PCN layer (`act[1]`), the layer that directly explains the senses.
- **Score:** G = Σ_h w_h (|e_h|₁ + intrinsic_w[0]·pe_h − social_weight·prox_h), with w_h ∝ discount^h normalized. `discount` can exceed 1 (5HT up to 2 gives 1.3), meaning the far future outweighs the near; the normalization keeps G's scale fixed.
- **Reflexes** (genome `reflex`, R = A·(1 + n_drives) = 70): a per-action bias plus a drive-error → action weight matrix, added to the logits. These are the inherited instincts.
- **intrinsic_w** has K = 1: the weight on expected prediction error in G (negative = curious, seeks the unknown).

### D41. Self-model inputs include the current body state and the efference copy
- `self_net(state)` sees [drives, body-frame velocity, nm, tanh(z), last action]. Without the current velocity it could never predict the next one, and an external shove was indistinguishable from normal motion (shoved 0.269 vs free 0.248). With it: 0.115 vs 0.032 (`test_distinguishes_agency`).
- Outputs: the next motor vector and the next body state [drives, velocity]. Lived: zero at birth, delta rule at `selfmodel_lr` × gate, no metaplasticity.

### D42. Bug fix: impact damage needs a speed threshold
- Pressing against a wall re-clamped a ~4 mm/s outward velocity every step, which the impact term turned into ≈ 0.2 integrity/s of continuous damage; it killed the organism within 30 s. Only collisions faster than `impact_threshold = 0.2` m/s now hurt.

### D43. Episodic memory built in step 5 (needed by the loop); storage details
- Key = all latent layers at the start of a trace (384 floats); value = the last `episodic_seq_len = 5` observations, which sleep replays as a sequence (SPEC 7.9 "through time"). Written every step from step 5 on; earlier windows are written with zero encoding.
- Age is write time plus a buffer clock, so `tick` is O(1) instead of O(capacity) per millisecond.
- Replay uses `jax.random.categorical` over salience logits (unwritten slots are −inf), which is equivalent to the SPEC's `choice(p=softmax(...))`.

### D44. Fitness task score is "health" against species reference setpoints
- `health = alive · (1 − mean|drives − Config.setpoint_init|)`. It is measured against fixed reference setpoints, not the evolved ones, so evolution cannot raise competence by moving its own goalposts. Used by fitness (`competence`, `robustness`) and the growth ratio.

### D45. I5 checks behavior; I3 implemented
- I5: NE, 5HT and OT must also change the action distribution (DA, ACh and CORT act through plasticity, precision and memory encoding).
- I3: no function in the package takes a mode flag, and the newborn's first step changes its PCN weights.

## Step 6 — memory/sleep.py, creature/life.py

### D46. What sleep consolidates and how
- **Replay unit:** a trace = the latent state at its start + 5 consecutive observations (D43). Sleep restores that latent state and lets the PCN re-live the sequence exactly as when awake (prior prediction, relaxation), and `replay_loss` is the mean prior surprise. `jax.grad` differentiates through the sequence (through time, via lateral weights and the persistent top latent) and through the relaxation iterations.
- **Optimizer:** plain SGD with global-norm clipping (`sleep_grad_clip = 1.0`). `sleep_lr = 1.0` after a scan on a lived organism (200 steps): 0.01 → 0.92× replayed surprise, 0.1 → 0.67×, 1.0 → 0.12×, with recent-experience surprise falling equally (0.13×), so nothing recent was sacrificed.
- **SPEC fix:** the SPEC's loop reuses one `rng` for every consolidation step (so it replays the same batch 200 times); the key is split per step.
- **Pruning:** |w| < `prune_threshold` in W_down and W_lat is set to 0; biases are not pruned. Pruned synapses can regrow through waking plasticity. Then W_up := W_down ᵀ exactly, resolving the slow re-alignment concern in D19.
- **Not consolidated:** the forward model and the self-model (the replay traces hold no action-outcome periods). The SPEC diagram's SLEEP → LAT edge is therefore only through the PCN latents.
- **Sleep takes no world time:** the world is paused while the organism consolidates. The metaplastic state and learning rates are untouched by sleep.

### D47. New file `creature/life.py` (not in the SPEC 6 layout)
- Schedules a life: `lax.scan` over waking steps, binned statistics, and sleep every `sleep_period` steps. It is shared by fitness (batched, static schedule) and handoff (one organism). It cannot live in `organism.py` without that module importing sleep, which I2's test forbids for the waking step.

## Step 7 — outer/genome.py, outer/fitness.py, outer/evolve.py

### D48. Genome encoding and expression
- Fields are SPEC 7.12's plus `da_tonic` (D31). Genes are stored unconstrained so one mutation scale suits all; `express` decodes them into the `Endowment`: `meta_tau` and `nm_tau` are log-encoded (exp, with the log clipped to ±20 so no constant is 0 or ∞), setpoints clipped to [0, 1], urgency to ≥ 0, `da_tonic` to [0, NM_MAX].
- Inherited synapses (`init_w`, `log_lr`, `meta_tau`, each (P,)) = PCN W_down, W_lat, biases, plus the forward model. W_up is not a gene: it is born as W_downᵀ (D19), and its rates and hardening are W_down's transposed.
- **Genome dimension: 276,234** = 3 × 92,046 synapses + 96 scalars (≤ 300k; SPEC 13 #2 low-rank not needed).
- `init_genome` (the prior) = Config birth defaults, random initial weights, `da_tonic ~ U(0.1, 1.0)`.

### D49. Fitness term definitions
- **Task score** is `organism.health` (D44); `competence` = last-quarter health. `competence at birth` (for the growth ratio) = first-quarter health of the same life.
- **robustness** = last-quarter health / health in the quarter before the body perturbation, clipped to [0, 1]. The perturbation, at dev_steps/2, scales mass, thrust and turn rate each by U(0.5, 1.5). Per-evaluation morphology randomization (SPEC 9.2) is already done by `randomize_world`.
- **curiosity** = mean learning progress, max(slow − fast running average of pred_err, 0). It is deliberately not multiplied by any gene, so a genome cannot inflate it by turning up a weight.
- **hack** = fraction of life with gate·lr < `statue_plasticity` + [mean homeostatic error < `trivial_err`].
- **Rejection** (SPEC 9.2): fitness := `reject_fitness` (−10) if plasticity is absent on > 90% of life, or if homeostatic error at birth is below `trivial_err`.
- Statistics are accumulated in bins of `fitness_bin_steps`; quarters are taken over bins. Developmental lives use `dev_episodic_capacity = 2048` because many lives run at once.

### D50. Evolution strategy (deviations from the SPEC sketch)
- Isotropic ES, as in SPEC 7.14's code. True CMA-ES needs a 276k × 276k covariance.
- **Mean step ∝ σ, not ∝ 1/σ:** `mean += es_lr · σ · (εᵀ r) / pop`. With the SPEC's `/ (pop · σ)` and an adaptive σ, the step grows as σ shrinks and the run diverged on a test landscape (improved for 80 generations, then degraded). `es_lr = 2.0` is CMA's c_m = 1 under centered ranks; parent fitness then rises monotonically in every 40-generation window.
- **σ adaptation:** the 1/5th success rule against the *parent* (the mean genome, evaluated each generation in the same worlds). Comparing against last generation's average candidate always gives ≈ 50% success, so σ grew without bound.
- **SPEC fix:** the sketch draws ε from the same `rng` every generation; keys are `fold_in(root, gen)`, which also makes checkpoint resume bit-identical.
- Common random numbers: all candidates and the parent share the generation's worlds. Candidates are evaluated with `lax.map` in batches of `eval_batch` to bound memory.

### D51. Invariant tests for whole-life modules
- I8 is tested as SPEC 2 states it, "enforced by fitness weighting": a genome inheriting a whole life's learned weights with plasticity off loses to the adaptive genome it came from, and every fitness evaluation reports a growth ratio.
- I10 vmap checks for sleep and fitness use batches of 8 and 2 (each element is a night of consolidation or whole lives); SPEC 10's 1024 is used for every per-step module.

## Step 8 — telemetry.py

### D52. Telemetry details
- `row(org, info)` produces exactly the SPEC 7.16 fields. `weight_norm` is over all PCN weights; `mean_meta` is the mean PCN metaplastic state; `lr_mean` is the mean effective PCN learning rate after hardening.
- Growth Ratio: competence = mean health (D44). Birth = the first `growth_window` (10⁴) steps; now = the latest complete window. It is NaN (undefined) until two windows exist, then clipped to [−1, 1] with the denominator floored at 10⁻⁶, so it is always defined and bounded once reported.
- `TelemetryStore` writes one `np.savez_compressed` file per flushed chunk and reads them back in order.

## Step 9 — fitness tuning (Phase 1e, reduced scale)

### D53. Competence = advantage over a resting twin (changes the meaning of "task score")
- **Finding:** with competence = health (D44), unevolved genomes scored Growth Ratio −0.04 to −0.09 and no weighting could raise it. Health measures *state*. A newborn starts healthy (birth drives are near the reference setpoints) and drifts, so "competence at birth" was starting luck, not skill, and GR was structurally ≤ 0.
- **Choice:** each life runs alongside a resting twin (`life.Twin`): the same world, the same world randomness and the same prediction-error stream, but the body always rests. Task score = health − twin health, i.e. what behavior adds. A flailing newborn scores < 0 (it burns energy and collides), a forager > 0. Used for `competence`, `competence_birth` and the growth ratio, in fitness (Phase 1) and telemetry (Phase 2). `robustness` stays on health, because a ratio of signed advantages is unstable.
- Consequence: a genome whose adult does no better than resting has competence ≤ 0 and growth ratio −1 (D49's clip). That is the correct verdict for a creature that never learned to do anything.

### D54. Presets and the tuning regime
- `config.preset("full")` is the SPEC configuration. `preset("tune")` is a reduced organism for CPU iteration: 8 rays, latent layers (32, 32), 2 relaxation iterations, 30 s lives, sleep every 10 s, 2 worlds, population 16.
- Tuning uses `metabolic_scale = 0.05`. At 0.01 (D38) a 30–60 s life never gets hungry, so resting is near-optimal and behavior barely matters (competence ≈ −0.03). At 0.05 unfed organisms starve within a minute (survival 0.78, competence ≈ −0.6), so foraging is a real, learnable skill.
- **Compute reality:** one full-preset evaluation is 8 lives × 200k steps; a generation is 513 of those; Phase 1f is 4,000 generations, on the order of 10¹² organism-steps, which needs the GPU-days SPEC 8 budgets. Only reduced-scale tuning is possible in this environment; results are in `runs/tune/evolve_tune.csv`.

## Step 10 — handoff.py

### D55. Phase 2 life
- `Life` = organism + world + resting twin + growth tracker. It is born once from `species_endowment.npz`, with `birth_seed` minted into the life key only in `handoff.main`. The birth world comes from its own `fold_in` stream.
- The life is stepped in `log_period` chunks with per-step telemetry rows (SPEC 7.16 fields + `growth_ratio` + `advantage`) and sleeps at multiples of `sleep_period`. The whole `Life` pytree is checkpointed every `checkpoint_period` steps and on exit; a restart restores it and continues at the same t (`test_continuity`: bit-identical state and telemetry).
- `run()` loads exactly two files: the genome, and, when resuming, its own checkpoint (`test_no_pretrained`). It imports nothing from `creature.outer` except `genome` (`test_handoff_no_evolution` checks `sys.modules` in a subprocess). The SPEC's "no call into outer/" is read as "no evolution or fitness": `Genome.load` is the SPEC's own interface. Presets moved to `config.py` so Phase 2 does not import `evolve.py`.

### D56. I9 entry points
- Only the `main()` of the two programs (`outer/evolve.py`, Phase 1; `handoff.py`, Phase 2) may mint a root key from a seed; the rest of those files, and every other module, must receive keys. `test_rng_explicit` enforces this per function.

### D57. RESULT of the first tuning run — evolution found the "dark room" (OPEN; needs a decision)
- **Run:** `preset("tune")`, 40 generations, population 16, 2 worlds × 30 s lives, seed 0 (~43 min on 4 CPU cores).
- **Fitness rose:** parent 4.10 → 5.86, best candidate 4.28 → 5.99. survival 1.0, robustness ≈ 0.99, learn_speed 0.51 → 0.88–1.0.
- **But competence stayed ≈ 0** (−0.03 … +0.009 against the resting twin), so the Growth Ratio is not meaningful (it flips among −1, 0 and 1 when competence ≈ 0). **It did not land in [0.40, 0.60].**
- **What evolved:** the organism rests 92% of the time (prior: 9%), mean speed 0.13 vs 0.84 m/s, and its prediction error falls to 0.0004. Stillness makes the world almost perfectly predictable, so `learn_speed`, the 3×-weighted term, approaches 1 without anything having been learned. This is the classic dark-room failure of surprise-minimizing agents, and the fitness currently pays for it.
- **Contributing causes, with proposed fixes:**
  1. *learn_speed rewards self-induced predictability.* Measure it relative to a naive predictor on the same stream (e.g. pred_err / error of "next obs = this obs"), or on a fixed probe sequence independent of behavior.
  2. *The twin shares the organism's prediction-error stream* (D53), so the boredom (novelty drive → 0) of a dark-room organism is cancelled in the advantage. Exclude novelty and safety from the twin comparison, or give the twin a fixed reference novelty.
  3. *30 s lives are too short for hunger* at `metabolic_scale = 0.05` (resting starves at ~200 s), so foraging is never needed. The SPEC's 200 s lives (`full` preset) would create the pressure but need far more compute.
- The full run (Phase 1f) should not start until these are decided: as configured, evolution would produce a statue that is still, not one that is rigid.

### D58. Fixes for the dark room (author approved the three proposals in D57)
1. **learn_speed is measured on a fixed probe, not on the organism's own experience.** At birth each world gets `probe_clips = 16` clips of `probe_clip_len = 20` consecutive frames, recorded by a scripted body that drives forward while sweeping its heading. At the end of every bin the PCN predicts each clip frame by frame from a blank latent state, as when awake (the first 5 frames give context only); `probe_err` is the mean surprise. learn_speed = relative fall of `probe_err`, first → last quarter.
   - The proposed alternative, normalizing by a "next = current" predictor, fails at a 1 ms timestep: frames change so little that the naive error is ~10⁻⁶ while the PCN's is ~10⁻², so the ratio is dominated by its floor.
   - A first probe that scored *single* frames by relaxation, without temporal context, got worse over life even for a learner (0.082 → 0.109). The waking PCN learns temporal prediction, mostly through lateral weights, so the probe has to measure that.
   - With clips (starvation disabled to isolate the signal, 200 s life): a wandering newborn's probe error falls 0.043 → 0.016 (learn_speed +0.57); the dark-room genome's rises 0.132 → 0.188 (−0.36). `test_stillness_does_not_score_learning` guards this.
2. **The twin no longer shares the organism's prediction error.** It gets a fixed reference surprise `twin_pred_err = 0.04` (≈ a naive newborn's), so an organism that bores itself (novelty → 0) is worse than the twin on novelty instead of cancelling out.
3. **Hunger within a life.** `metabolic_scale` 0.01 → 0.07, so a resting body starves in ≈ 143 s, inside the SPEC's 200 s life; an active, unfed newborn in ≈ 30–40 s. The `tune` preset now uses full 200 s lives (sleep every 50 s). The shrunken test config keeps 0.01, so mechanism tests run long lives without starving.
- **Bug fix:** the Growth Ratio returned +1 when final competence was 0 and birth competence negative (0 − negative, divided by the ε floor). It is now −1 whenever final competence is not positive, in both fitness and telemetry.
- **Under the new fitness** (2 worlds, 200 s): unevolved prior −0.24 (survival 0.16), old dark-room genome −0.33 (survival 0.62, learn_speed −0.25). Stillness no longer wins; the selection pressure is now foraging.

### D59. RESULT of the second tuning run (after D58) — learning and survival rise; no foraging yet (OPEN)
- **Run:** `preset("tune")` with 200 s lives, 10 generations, population 16, 2 worlds, seed 0 (75 min). Log: `runs/tune2/evolve_tune.csv`.
- **Moving the right way:** parent fitness −0.43 → 1.39 (best candidate 0.24 → 2.84 at gen 8); survival 0.14 → 0.47; probe learn_speed 0.14 → 0.35. Unlike D57, learning rises together with fitness: the dark room no longer pays.
- **What evolved:** lifespan 29 s → 63 s (mean of 3 worlds), mostly by rationing energy: it rests 61% of the time (prior 9%) and grabs 3% (prior 20%). It still almost never eats (0.1 s of eating, 0.10 energy in a life).
- **Growth Ratio still −1:** every organism, and the resting twin, is dead before the last quarter, so final competence = 0. The 0.40–0.60 band cannot be reached until organisms survive the whole life.
- **Why foraging is not found:** food is sparse (3 patches of ~0.3 m in a 10 × 10 m arena, eaten only within 0.3 m reach while grabbing), the organism has no sense that points to food beyond line of sight (8 rays at this preset), and 10 generations × 16 genomes is a very small search. Options: more or larger food patches (a sandbox parameter, not SPEC-fixed); an olfactory gradient sense (a low-level feature, allowed by SPEC 3); a longer and larger search (Phase 1f compute); or a staged curriculum (start with less metabolic pressure).

### D60. Making foraging learnable (author's choice of options 1 and 2 from D59)
- **More, larger food:** `n_food` 3 → 6, patch radius 0.25–0.35 → 0.4–0.6 m. These are sandbox parameters; SPEC 7.3 says only "a few movable objects".
- **Smell:** two odor sensors ("nostrils") on the body surface at ±0.8 rad from the heading report Σ stock · exp(−distance / 2 m) over food patches. It is a raw chemical intensity (no label, no coordinate), allowed as a low-level feature by SPEC 3. Its left−right difference points toward food out of sight, and depleted patches stop smelling. Observation grows by 2 (79 dims at the SPEC preset); genome dimension is now **277,560** (≤ 300k).
- Test adjustments caused by richer food: `test_learn_speed_dominates` now uses 20 s lives (probe-measured learning does not show within the shrunken config's 4 s); `test_distinguishes_agency`'s margin is 1.5× instead of my earlier 2×, since eating now moves the energy drive in ways the young self-model has not learned (measured 1.9×; SPEC 7.11 only requires "higher").

### D61. RESULT of the third tuning run (after D60) — foraging appears; organisms still starve (OPEN)
- **Run:** `preset("tune")`, 10 generations, population 16, 2 worlds, seed 0 (70 min). Log: `runs/tune3/evolve_tune.csv`.
- Parent fitness −0.64 → 0.39 (best candidate up to 2.56). The per-report evaluations of the mean genome are noisy (2 fresh worlds each): at generation 6 survival was 0.76 and competence turned positive (0.37) for the first time in any run; at generations 8 and 10 survival was 0.39 and 0.17.
- **Behavior of the generation-10 genome over 6 worlds:** it now eats in 4 of 6 worlds (1.2–2.2 energy units per life, more than a full store), against near zero in run 2 (D59). It no longer rations (rests 12%, prior 10%; run 2 rested 61%). Lifespan 33–78 s (mean 54 s vs prior 35 s).
- **All six lives still end in starvation** (integrity 0.54–0.99 at death). Food is not scarce: 6 units of initial stock plus regrowth far exceed its burn of ≈ 0.04 units/s. The organism does not return to food reliably, which is exactly the skill lifetime learning should improve, and the Growth Ratio cannot be measured until lives reach the last quarter.
- **Next levers:** a larger, longer search (population and generations; Phase 1f compute); evaluation noise (more worlds per genome, or fixed evaluation worlds for the mean-genome reports); or a metabolic curriculum (start gentler so lives are long enough to learn, then tighten).

### D62. Metabolic curriculum and fixed report worlds (Phase 1e)
- `evolve(..., curriculum=f)` takes a per-generation config (one compile per distinct stage; resume stays bit-identical because the config is a pure function of the generation). The CLI's `--curriculum 'scale:gens,...'` schedules `metabolic_scale`: early generations live long enough to learn foraging, later ones face the target pressure (D61's "eats, but not enough").
- Mean-genome reports are now always evaluated in the same `--report-worlds` (default 4) worlds, at the preset's target metabolism, so report-to-report changes reflect the genome, not the draw of worlds (D61's survival swinging 0.76 → 0.17).

### D63. Sensory reflex arcs (chemotaxis)
- **Gap found:** inherited reflexes mapped only interoception (drive error) to actions, so evolution had no way to encode an instinct like "turn toward the stronger smell". Learning to follow an odor gradient from scratch is hard for the forward model: its horizon is 4 × 100 ms and the reward (eating) is rare and delayed.
- **Choice:** reflex arcs also read the two raw odor sensors: per action a bias, a weight from each drive error, and a weight from each nostril. R = 10 × (1 + 6 + 2) = 90 genes. Genome dimension 277,580. SPEC 7.12 lists "reflex arcs" as inherited; a stimulus→response arc on a raw sense is the classic form (taxis), and it encodes no knowledge of any particular world, only a disposition, consistent with I8.

### D64. RESULT of the fourth tuning run (curriculum, D62) — best so far; Growth Ratio still not measurable in most worlds
- **Run:** `preset("tune")`, 20 generations, curriculum 0.015 (6) → 0.03 (6) → 0.05 (4) → 0.07 (4), population 16, 2 worlds per candidate; reports in 4 fixed worlds at 0.07. 2 h 46 min. Log: `runs/tune4/evolve_tune.csv`. (Run before D63's chemotaxis reflex.)
- **At the target metabolism:** survival 0.22 → 0.51 (gen 18) / 0.45 (gen 20); probe learn_speed −0.13 → 0.30, rising steadily (the highest yet); competence > 0 in one of the 4 report worlds at gen 18 (0.08, Growth Ratio −0.5 averaged over worlds).
- Most lives still end before the last quarter, so competence is 0 there and the Growth Ratio is −1. The next run continues from this genome with D63's sensory reflex genes added (initialized to 0), at the target metabolism.

### D65. Growth Ratio aggregated over worlds as a ratio of mean competences (bug fix)
- Fitness averaged per-world growth ratios, each clipped to [−1, 1]. With one world where the organism survives (ratio clipped to +1) and three where it starved (−1), the report read −0.5, which is meaningless. The ratio is now computed from competence averaged over worlds (SPEC 9.3 defines it on competence). `growth_ratio_raw` (unclipped) is also reported, because clipping at 1 hides how far into "blank slate" territory a genome is.
- Under the twin baseline (D53) a newborn's competence is ≤ 0 (it does worse than resting), so any positive final competence gives a raw ratio > 1. SPEC 9.3: "≈ 1 → blank slate with an expensive head start. Raise w_competence." Reaching 0.40–0.60 needs instincts that make newborns better than resting, which is the lever once lives last to the final quarter.

### D66. RESULT of the fifth tuning run (warm start + chemotaxis, D63) and the first w_competence adjustment
- **Run:** 20 generations at the target metabolism, warm-started from run 4's genome with zeroed sensory-reflex genes. 2 h 39 min. Log: `runs/tune5/evolve_tune.csv`.
- **Final mean genome (4 fixed report worlds):** survival 0.68 (run 4 ended at 0.45), competence vs twin 0.29, probe learn_speed 0.43 (highest yet), robustness 0.38. Competence at birth −0.046, so the raw growth ratio is (0.29 + 0.046) / 0.29 ≈ 1.16: the "blank slate" end of SPEC 9.3.
- **Behavior over 6 fresh worlds:** full 200 s lives in 2 of 6, eating 1.7 and 4.6 energy units; in the other 4 it eats almost nothing and dies at ≈ 80 s. The prior lives 28–52 s everywhere. Foraging skill exists but does not yet generalize across worlds.
- **Phase 2 end to end** with this genome (`python -m creature.handoff --preset tune --runs runs/tune5`): one birth, a continuous life of 277 s with 5 sleeps, per-step telemetry and checkpoints, death by starvation, final growth ratio +1.
- **Adjustment (SPEC 9.3: "≈ 1 → raise w_competence"):** `TUNED = {w_competence: 3.0}` applied to both presets. `Config()` keeps the SPEC defaults; the presets carry tuning results, so the full run inherits them.

### D67. Competence at birth = an instinct-only newborn in the adult's situation (measurement fix)
- **Problem:** under the twin baseline, first-quarter competence is always slightly negative (any activity costs energy while the resting twin's drives are still near reference). The growth ratio was therefore pinned at ≥ 1 regardless of the genome, and raising `w_competence` (D66) could not move it; round 6 (w_competence = 3) did worse and was stopped at generation 16.
- **Choice:** at 3/4 of each developmental life the adult continues for the last quarter, and in parallel a newborn from the same genome takes its place: the same world state, body state, twin and step keys, with every waking learning rate at the floor (`fitness.frozen`). competence_birth = the newborn's advantage, competence = the adult's, growth ratio = (adult − newborn) / adult, i.e. one task scored at two ages. The newborn is a Phase 1 measuring instrument, never the living organism, so I3 (the organism has no non-learning mode) is untouched. Cost ≈ +25% per evaluation.
- **Validation:** a genome with awake plasticity off scores growth ratio 0.04 (statue → ≈ 0). A newborn that is allowed to learn during the scored quarter catches up within it (forward-model learning is fast), which is why the probe must be frozen.
- **Diagnosis of the round-5 genome** (4 fixed worlds): instinct-only newborn 0.32 vs adult 0.29 (0.38 vs 0.39 without the mid-life body perturbation). Its competence is innate; **its growth ratio is ≈ 0: a statue.** Freezing the forward model makes adult = newborn; freezing the PCN collapses behavior. The adult's deficit comes from the body perturbation: its synapses hardened on the old body.

### D68. Ambiguous smell: the hazard also smells, by a per-world amount
- **Why the statue won:** food odor meant food in every world, so an inherited "turn toward the stronger smell, grab when it is strong" instinct solved foraging everywhere, and nothing required learning. SPEC 9.2 asks for selection for *general learnability* across randomized worlds.
- **Choice:** the hazard emits odor with strength U(0, 1.5) relative to a full food patch, drawn per world (`hazard_odor_range`). The organism must learn, in its own world, what its smells lead to (by vision, by experience).
- **Effect on the round-5 genome:** competence 0.29 → 0.08; the instinct no longer suffices.
- `TUNED.w_competence` back to 1.0 (SPEC 9.3: growth ratio ≈ 0 → lower w_competence).

### D69. RESULT of the seventh tuning run (D67 measurement + D68 ambiguous smell) — the growth ratio reached the target band, then was lost
- **Run:** 20 generations at the target metabolism, warm-started from run 5, population 16, 2 worlds per candidate; reports in 4 fixed worlds. ~3 h. Log: `runs/tune7/evolve_tune.csv`.
- **Growth ratio by report:** 0.42, 0.08, 0.15, 0.08, **0.55** (gen 8), **0.59** (gen 10), 0.35, −0.20, 0.05, 0.37, −1.0 (gen 20).
- **Generation 10** was the best on every term: growth ratio **0.59** (inside SPEC 9.3's 0.40–0.60), adult competence 0.59 vs an instinct-only newborn's 0.25 in the same situation, survival 0.87, robustness 0.74, probe learn_speed 0.26.
- **Not held:** later generations drifted back to instinct (gen 14: newborn 0.58 > adult 0.49) and then collapsed (gen 20: survival 0.50, competence 0). With 16 genomes per generation the search finds the band but cannot stay in it. A larger population (the full preset's 512) and more evaluation worlds are the expected remedy; they are Phase 1f compute.
- The runner kept only the latest genome; it now saves every reported genome. Evolution is bit-reproducible, so generation 10 is being regenerated by replay (`runs/tune7_replay`).

### D70. The species endowment, generalization check, and the Phase 2 life
- Generation 10 of run 7 was regenerated by replay (`runs/tune7_replay`); every generation's fitness and the generation-10 report matched the original run bit for bit (growth ratio 0.5865).
- **Generalization:** in 16 worlds never used for selection (two sets of 8), growth ratio **0.31 / 0.36**, survival 0.60 / 0.59, adult competence 0.17 / 0.32 vs instinct-only newborn 0.11 / 0.20. The 0.59 was partly selection luck on the 4 fixed report worlds. About a third of final competence is learned in life: clearly not a statue (≈ 0) and not a blank slate (≈ 1), but below the 0.40–0.60 band.
- This genome is the species endowment (`runs/species/species_endowment.npz`), and the Phase 2 life was started from it with the `tune` preset (`python -m creature.handoff --preset tune --runs runs/species --life runs/life`). It lives until it dies or this ephemeral session ends; its checkpoint and telemetry are in `runs/life/`.
- **What remains for the SPEC's finish line:** Phase 1f at full scale (population 512, 8 worlds, the full organism; GPU; README) to make the band hold and generalize, then a Phase 2 life on a machine that stays up for weeks.
