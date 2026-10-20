"""All hyperparameters for PROJECT CREATURE (SPEC 7.1).

Every magic number in the codebase lives here; module bodies take a `Config`
instead of using literals. The dataclass is frozen so it is hashable and can be
passed to `jax.jit` as a static argument without being traced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    """Frozen global configuration. Fields map one-to-one to SPEC 7.1."""

    dt: float = 1e-3  # simulation timestep in seconds (1 ms real-time clock, SPEC 5.1)
    n_drives: int = 6  # number of homeostatic drives (SPEC 7.2)
    n_neuromod: int = 6  # number of neuromodulator channels: DA, 5HT, NE, ACh, OT, CORT (SPEC 7.6)
    layer_sizes: tuple[int, ...] = (64, 128, 128, 64)  # predictive-coding layer widths, bottom to top (SPEC 7.4)
    n_fast_synapses: int = 0  # fast-weight synapse count; 0 until the Phase 2 addition (SPEC 7.5)
    episodic_capacity: int = 100_000  # episodic ring-buffer slots (SPEC 7.8)
    sleep_period: int = 60_000  # awake steps between sleep episodes (SPEC 7.9)
    sleep_replay_batch: int = 512  # traces replayed per consolidation step (SPEC 7.9)
    sleep_consolidation_steps: int = 200  # consolidation passes per sleep episode (SPEC 7.9)
    pop_size: int = 512  # evolution population size per generation (SPEC 7.14)
    n_eval_worlds: int = 8  # randomized worlds each genome is evaluated in (SPEC 7.13, 9.2)
    dev_steps: int = 200_000  # lifetime steps per developmental fitness rollout (SPEC 7.13)
    generations: int = 4000  # evolution generations in the full Phase 1f run (SPEC 7.14)
    # fitness weights — see 9.1
    w_survival: float = 1.0  # weight on fraction of dev_steps alive (SPEC 9.1)
    w_learn: float = 3.0  # weight on relative early→late pred_err reduction; the anti-statue term (SPEC 9.1)
    w_competence: float = 1.0  # weight on mean task score in the final quarter (SPEC 9.1)
    w_robustness: float = 2.0  # weight on recovery after body perturbation (SPEC 9.1)
    w_curiosity: float = 0.5  # weight on mean intrinsic reward; the anti-plateau term (SPEC 9.1)
    w_hack: float = 0.5  # penalty weight on exploit magnitude, subtracted (SPEC 9.1)

    # --- sandbox: layout and senses (SPEC 7.3) ---
    arena_half_width: float = 5.0  # half side of the square arena in metres; walls at ±this
    spawn_margin: float = 0.5  # minimum distance from walls for anything placed at spawn
    n_objects: int = 4  # movable inert objects per world
    n_food: int = 3  # regrowing food patches per world (the only energy source)
    n_rays: int = 16  # egocentric vision rays
    ray_fov: float = 4.0  # total vision field of view in radians, centred on heading
    ray_range: float = 6.0  # max ray distance in metres; beyond it a ray sees nothing
    act_dim: int = 4  # motor vector: [thrust, turn, manipulate, vocalize], each in [-1, 1]
    # entity base colours (RGB) the rays report; per-world jitter is added on top
    color_wall: tuple[float, float, float] = (0.5, 0.5, 0.5)  # arena walls
    color_object: tuple[float, float, float] = (0.2, 0.3, 0.8)  # movable objects
    color_food: tuple[float, float, float] = (0.2, 0.8, 0.2)  # food patches, dimmed as they deplete
    color_caregiver: tuple[float, float, float] = (0.9, 0.7, 0.3)  # caregiver agent
    color_hazard: tuple[float, float, float] = (0.9, 0.1, 0.1)  # hazard
    color_jitter: float = 0.1  # std of per-world Gaussian colour jitter
    food_dim_floor: float = 0.3  # brightness of an empty food patch relative to a full one

    # --- sandbox: per-world randomization ranges, (lo, hi) uniform (SPEC 7.3, 9.2) ---
    friction_range: tuple[float, float] = (0.5, 2.0)  # linear velocity damping, 1/s
    agent_mass_range: tuple[float, float] = (0.8, 1.2)  # body mass in kg (morphology jitter)
    agent_radius_range: tuple[float, float] = (0.18, 0.22)  # body radius in m (morphology jitter)
    max_thrust_range: tuple[float, float] = (3.0, 5.0)  # force at full thrust, N
    max_turn_range: tuple[float, float] = (2.5, 3.5)  # turn rate at full turn, rad/s
    obj_mass_range: tuple[float, float] = (0.5, 3.0)  # movable object mass, kg
    obj_radius_range: tuple[float, float] = (0.15, 0.35)  # movable object radius, m
    food_radius_range: tuple[float, float] = (0.25, 0.35)  # food patch radius, m
    food_regrow_range: tuple[float, float] = (0.01, 0.05)  # food regrowth, capacity units/s
    hazard_radius_range: tuple[float, float] = (0.4, 0.8)  # hazard radius, m
    cg_gain_range: tuple[float, float] = (0.5, 2.0)  # caregiver approach gain toward preferred distance, 1/s
    cg_pref_dist_range: tuple[float, float] = (0.5, 1.5)  # caregiver preferred distance from the organism, m
    cg_wander_range: tuple[float, float] = (0.1, 0.5)  # caregiver wander-velocity noise scale, m/s
    cg_voc_response_range: tuple[float, float] = (0.0, 2.0)  # extra approach speed per unit vocalization, m/s
    cg_max_speed_range: tuple[float, float] = (0.5, 1.5)  # caregiver speed cap, m/s
    cg_warmth_range: tuple[float, float] = (0.0, 0.3)  # ambient warmth the caregiver adds at zero distance
    ambient_base_range: tuple[float, float] = (0.3, 0.7)  # ambient temperature at arena centre, [0, 1]
    ambient_amp_range: tuple[float, float] = (0.1, 0.5)  # ambient change from centre to wall along the gradient

    # --- sandbox: fixed physics (SPEC 7.3) ---
    cg_radius: float = 0.25  # caregiver body radius, m
    cg_wander_theta: float = 1.0  # mean-reversion rate of caregiver wander velocity, 1/s
    warmth_scale: float = 1.0  # distance over which caregiver warmth falls by 1/e, m
    reach: float = 0.3  # manipulation reach beyond the body surface, m
    grasp_k: float = 5.0  # spring gain pulling a grasped object toward the mouth, 1/s^2
    food_capacity: float = 1.0  # maximum food held by a patch, in energy units
    ingest_rate: float = 1.0  # energy gained per second while fully manipulating a stocked patch
    hazard_damage_rate: float = 0.5  # integrity lost per second fully inside the hazard
    impact_damage_gain: float = 0.05  # integrity lost per m/s of collision speed above the threshold
    impact_threshold: float = 0.2  # m/s; slower contact (resting, pushing against a wall) does no damage
    proximity_scale: float = 1.0  # distance over which caregiver proximity falls by 1/e, m

    # --- homeostat (SPEC 7.2) ---
    metabolic_cost_rest: float = 0.1  # energy/s at rest (SPEC METABOLIC_COST['rest'])
    metabolic_cost_locomote: float = 0.5  # energy/s at full locomotion (SPEC METABOLIC_COST['locomote'])
    metabolic_cost_manipulate: float = 0.8  # energy/s at full manipulation (SPEC METABOLIC_COST['manipulate'])
    metabolic_cost_vocalize: float = 0.3  # energy/s at full vocalization (SPEC METABOLIC_COST['vocalize'])
    metabolic_scale: float = 0.01  # multiplies every metabolic cost; SPEC rates starve an active newborn in ~2 s (DECISIONS D38)
    thermal_k: float = 0.2  # body-temperature relaxation rate toward ambient, 1/s
    thermal_action_gain: float = 0.05  # body heat produced per second at full locomotion
    repair_rate: float = 0.01  # passive integrity repair per second
    social_k: float = 0.5  # social-drive relaxation rate toward caregiver proximity, 1/s
    novel_k: float = 0.1  # novelty relaxation rate toward information intake, 1/s (boredom timescale)
    safety_k: float = 0.2  # safety relaxation rate toward its target, 1/s
    safety_var_gain: float = 4.0  # scales pred-err variance so safety can reach 0 (variance of [0,1] values is ≤ 0.25)
    info_gain: float = 10.0  # scales mean recent pred-err into information intake; naive-PCN surprise (~0.05) maps to ~0.5
    pred_err_window: int = 256  # steps of prediction error kept for novelty and safety
    drive_init: tuple[float, ...] = (1.0, 0.5, 1.0, 0.5, 0.5, 1.0)  # drives at birth, in DRIVES order

    # --- predictive coding (SPEC 7.4) ---
    pcn_nonlinearity: str = "tanh"  # value-node nonlinearity applied before weights: "tanh" or "linear"
    pcn_inference_steps: int = 5  # relaxation iterations per timestep
    pcn_inference_rate: float = 0.1  # value-node step size per relaxation iteration
    pcn_top_prior: float = 0.1  # leak of the free top layer toward 0 per iteration (Gaussian prior)
    pcn_init_scale: float = 1.0  # birth W_down std is this / sqrt(fan_in); W_lat and biases start at 0
    pcn_weight_decay: float = 0.01  # 1/s decay on W_down and W_up; makes W_up re-align to W_down^T (Kolen–Pollack)

    # --- plasticity (SPEC 7.5); init_* are birth defaults the genome later evolves; rule constants live in substrate/plasticity.py ---
    init_log_lr: float = 0.0  # per-synapse log learning rate at birth (lr = 1/s, the clip ceiling)
    init_meta_tau: float = 0.1  # per-synapse metaplasticity hardening constant at birth

    # --- neuromodulation (SPEC 7.6); channel constants live in substrate/neuromod.py ---
    nm_gain_init: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # birth driver gains, evolved
    nm_tau_init: tuple[float, ...] = (0.2, 5.0, 0.1, 0.5, 10.0, 20.0)  # birth time constants in s, evolved
    da_tonic_prior: tuple[float, float] = (0.1, 1.0)  # uniform prior on the evolved tonic DA level (genome da_tonic)

    # --- affect (SPEC 7.7); birth defaults the genome later evolves ---
    setpoint_init: tuple[float, ...] = (0.8, 0.5, 1.0, 0.6, 0.4, 0.8)  # preferred drive levels at birth, DRIVES order
    urgency_init: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # weight of each drive's error at birth

    # --- policy: active inference over motor primitives (SPEC 7.10) ---
    motor_primitives: tuple[tuple[float, float, float, float], ...] = ((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (-0.5, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0), (1.0, 0.5, 0.0, 0.0), (1.0, -0.5, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.3, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))  # candidate actions: rest, fwd, back, left, right, fwd-left, fwd-right, grab, approach-grab, vocalize
    policy_period: int = 100  # steps a chosen primitive is held (100 ms); also the forward model's prediction step
    policy_horizon: int = 4  # forward-model rollout depth, in policy periods
    dyn_interaction_dims: int = 8  # latent dims crossed with the action one-hot in forward-model features
    n_intrinsic: int = 1  # intrinsic weights: [epistemic value of expected prediction error]
    progress_tau_fast: float = 1.0  # s, fast running average of pred_err for learning progress
    progress_tau_slow: float = 10.0  # s, slow running average; progress = max(slow − fast, 0)

    # --- self-model (SPEC 7.11); lived, never inherited ---
    selfmodel_lr: float = 1.0  # learning rate per second, scaled by the plasticity gate

    # --- episodic memory (SPEC 7.8) ---
    episodic_seq_len: int = 5  # observations per trace; sleep replays them as a sequence
    episodic_sal_decay: float = 1e-3  # 1/s decay of salience with trace age
    episodic_temp: float = 1.0  # softmax temperature of salience-weighted replay

    # --- sleep (SPEC 7.9) ---
    sleep_lr: float = 1.0  # SGD step size for consolidation (scan: 0.01 → 0.92x surprise, 0.1 → 0.67x, 1.0 → 0.12x after 200 steps)
    sleep_grad_clip: float = 1.0  # global gradient-norm clip per consolidation step
    prune_threshold: float = 1e-4  # synapses with |w| below this are removed

    # --- fitness (SPEC 7.13, 9) ---
    fitness_bin_steps: int = 1000  # trajectory statistics are accumulated in bins of this many steps
    dev_episodic_capacity: int = 2048  # episodic capacity during developmental evaluation (memory-bound under vmap)
    perturb_scale: tuple[float, float] = (0.5, 1.5)  # body perturbation: multiplicative range on mass, thrust, turn
    reject_fitness: float = -10.0  # fitness assigned to a rejected (degenerate) genome (SPEC 9.2)
    statue_plasticity: float = 1e-3  # gate·lr below this counts as a step with no plasticity
    statue_fraction: float = 0.9  # reject if plasticity is absent on more than this fraction of steps
    trivial_err: float = 0.05  # reject if total homeostatic error at birth is below this (setpoints trivially met)

    # --- evolution (SPEC 7.14) ---
    es_sigma_init: float = 0.05  # initial mutation scale
    es_lr: float = 2.0  # mean step = es_lr · sigma · mean(rank · noise); ≈ CMA c_m = 1 under centered ranks
    es_sigma_bounds: tuple[float, float] = (1e-3, 0.5)  # sigma is clipped to this range
    es_success_target: float = 0.2  # 1/5 success rule: target fraction of candidates beating the last mean fitness
    es_sigma_damping: float = 10.0  # damping of the success-rule sigma update

    # --- telemetry and Phase 2 (SPEC 7.15, 7.16) ---
    growth_window: int = 10_000  # steps per competence estimate; growth ratio is logged this often
    checkpoint_period: int = 100_000  # Phase 2 steps between life-state checkpoints
    log_period: int = 1_000  # Phase 2 steps between telemetry flushes
    birth_seed: int = 0  # the single seed of the one Phase 2 life


def preset(name: str) -> Config:
    """Named configurations. `full` is the SPEC configuration (Phase 1f needs a GPU and days).
    `tune` is a reduced organism with short lives, for Phase 1e iteration on a CPU (DECISIONS D54)."""
    if name == "full":
        return Config()
    if name == "tune":
        return replace(
            Config(), n_rays=8, layer_sizes=(32, 32), pcn_inference_steps=2, dev_steps=30_000, sleep_period=10_000,
            sleep_replay_batch=64, sleep_consolidation_steps=50, fitness_bin_steps=1_000, n_eval_worlds=2,
            dev_episodic_capacity=1_024, pop_size=16, metabolic_scale=0.05,
        )
    raise ValueError(f"unknown preset {name!r}")
