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
