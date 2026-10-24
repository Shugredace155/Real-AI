"""2D vectorized physics sandbox (SPEC 7.3).

A square walled arena with a few movable objects, regrowing food patches, one
scripted caregiver and one hazard. Everything is a disk; walls are the arena
boundary. The organism perceives the world only through egocentric rays that
report depth and colour of the first surface hit, plus proprioception and its
own interoceptive vector, plus a two-nostril food odor — never labels or
coordinates (SPEC 3, 7.3).

Every function is pure and operates on a single world; batching is done by the
caller with `jax.vmap` (Invariant I10). All randomness enters through explicit
PRNG key arguments (Invariant I9).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from creature.config import Config

# Entity-type indices, used only to look up per-type colours.
WALL, OBJECT, FOOD, CAREGIVER, HAZARD = range(5)

# Motor-vector component indices (Config.act_dim == 4).
THRUST, TURN, MANIPULATE, VOCALIZE = range(4)


@struct.dataclass
class WorldParams:
    """Per-world physics, morphology and caregiver policy, drawn by `randomize_world`."""

    friction: jnp.ndarray  # () linear damping, 1/s
    agent_mass: jnp.ndarray  # ()
    agent_radius: jnp.ndarray  # ()
    max_thrust: jnp.ndarray  # ()
    max_turn: jnp.ndarray  # ()
    obj_mass: jnp.ndarray  # (K,)
    obj_radius: jnp.ndarray  # (K,)
    food_radius: jnp.ndarray  # (F,)
    food_regrow: jnp.ndarray  # ()
    hazard_radius: jnp.ndarray  # ()
    cg_gain: jnp.ndarray  # ()
    cg_pref_dist: jnp.ndarray  # ()
    cg_wander: jnp.ndarray  # ()
    cg_voc_response: jnp.ndarray  # ()
    cg_max_speed: jnp.ndarray  # ()
    cg_warmth: jnp.ndarray  # ()
    ambient_base: jnp.ndarray  # ()
    ambient_amp: jnp.ndarray  # ()
    ambient_dir: jnp.ndarray  # (2,) unit vector of the temperature gradient
    colors: jnp.ndarray  # (5, 3) RGB per entity type


@struct.dataclass
class World:
    """Full sandbox state. `ingest`, `damage`, `ambient`, `proximity` and `impact`
    are the body-contact signals of the most recent step, read by the homeostat."""

    agent_pos: jnp.ndarray  # (2,)
    agent_vel: jnp.ndarray  # (2,)
    agent_heading: jnp.ndarray  # ()
    obj_pos: jnp.ndarray  # (K, 2)
    obj_vel: jnp.ndarray  # (K, 2)
    food_pos: jnp.ndarray  # (F, 2)
    food_amount: jnp.ndarray  # (F,)
    cg_pos: jnp.ndarray  # (2,)
    cg_wvel: jnp.ndarray  # (2,) caregiver wander velocity (Ornstein–Uhlenbeck state)
    hazard_pos: jnp.ndarray  # (2,)
    last_action: jnp.ndarray  # (act_dim,)
    ingest: jnp.ndarray  # () energy intake rate this step
    damage: jnp.ndarray  # () integrity damage rate this step
    impact: jnp.ndarray  # () collision speed absorbed by the body this step
    ambient: jnp.ndarray  # () ambient temperature at the body
    proximity: jnp.ndarray  # () caregiver proximity in (0, 1]
    params: WorldParams


def effectors(action: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Decode a motor vector into effector drives.

    Args:
        action: (act_dim,) motor vector; values are clipped to [-1, 1].
    Returns:
        (thrust, turn, locomote, manipulate, vocalize). thrust/turn are signed;
        locomote, manipulate, vocalize are intensities in [0, 1]. A non-positive
        manipulate/vocalize component means that effector is off.
    Invariant: I10 (pure, elementwise).
    """
    a = jnp.clip(action, -1.0, 1.0)
    thrust, turn = a[THRUST], a[TURN]
    locomote = jnp.maximum(jnp.abs(thrust), jnp.abs(turn))
    manipulate = jnp.maximum(a[MANIPULATE], 0.0)
    vocalize = jnp.maximum(a[VOCALIZE], 0.0)
    return thrust, turn, locomote, manipulate, vocalize


def _uniform(key: jax.Array, rng: tuple[float, float], shape: tuple[int, ...] = ()) -> jnp.ndarray:
    return jax.random.uniform(key, shape, minval=rng[0], maxval=rng[1])


def randomize_world(key: jax.Array, cfg: Config) -> World:
    """Draw a world with randomized layout, physics, morphology and caregiver policy.

    Args:
        key: PRNG key; the world is a deterministic function of it.
        cfg: global config.
    Returns:
        A freshly spawned `World`.
    Invariants: I9 (randomness only from `key`), I10 (vmappable over keys).
    """
    ks = iter(jax.random.split(key, 32))
    lim = cfg.arena_half_width - cfg.spawn_margin
    spot = lambda shape: jax.random.uniform(next(ks), shape + (2,), minval=-lim, maxval=lim)

    theta = jax.random.uniform(next(ks), (), minval=-jnp.pi, maxval=jnp.pi)
    base = jnp.array(
        [cfg.color_wall, cfg.color_object, cfg.color_food, cfg.color_caregiver, cfg.color_hazard]
    )
    colors = jnp.clip(base + cfg.color_jitter * jax.random.normal(next(ks), base.shape), 0.0, 1.0)

    params = WorldParams(
        friction=_uniform(next(ks), cfg.friction_range),
        agent_mass=_uniform(next(ks), cfg.agent_mass_range),
        agent_radius=_uniform(next(ks), cfg.agent_radius_range),
        max_thrust=_uniform(next(ks), cfg.max_thrust_range),
        max_turn=_uniform(next(ks), cfg.max_turn_range),
        obj_mass=_uniform(next(ks), cfg.obj_mass_range, (cfg.n_objects,)),
        obj_radius=_uniform(next(ks), cfg.obj_radius_range, (cfg.n_objects,)),
        food_radius=_uniform(next(ks), cfg.food_radius_range, (cfg.n_food,)),
        food_regrow=_uniform(next(ks), cfg.food_regrow_range),
        hazard_radius=_uniform(next(ks), cfg.hazard_radius_range),
        cg_gain=_uniform(next(ks), cfg.cg_gain_range),
        cg_pref_dist=_uniform(next(ks), cfg.cg_pref_dist_range),
        cg_wander=_uniform(next(ks), cfg.cg_wander_range),
        cg_voc_response=_uniform(next(ks), cfg.cg_voc_response_range),
        cg_max_speed=_uniform(next(ks), cfg.cg_max_speed_range),
        cg_warmth=_uniform(next(ks), cfg.cg_warmth_range),
        ambient_base=_uniform(next(ks), cfg.ambient_base_range),
        ambient_amp=_uniform(next(ks), cfg.ambient_amp_range),
        ambient_dir=jnp.array([jnp.cos(theta), jnp.sin(theta)]),
        colors=colors,
    )
    zero = jnp.zeros(())
    world = World(
        agent_pos=spot(()),
        agent_vel=jnp.zeros(2),
        agent_heading=jax.random.uniform(next(ks), (), minval=-jnp.pi, maxval=jnp.pi),
        obj_pos=spot((cfg.n_objects,)),
        obj_vel=jnp.zeros((cfg.n_objects, 2)),
        food_pos=spot((cfg.n_food,)),
        food_amount=jnp.full((cfg.n_food,), cfg.food_capacity),
        cg_pos=spot(()),
        cg_wvel=jnp.zeros(2),
        hazard_pos=spot(()),
        last_action=jnp.zeros(cfg.act_dim),
        ingest=zero,
        damage=zero,
        impact=zero,
        ambient=zero,
        proximity=zero,
        params=params,
    )
    # Fill the body-contact signals so the first homeostat step sees a consistent world.
    return world.replace(ambient=_ambient_at(world, cfg), proximity=_proximity(world, cfg))


def _ambient_at(world: World, cfg: Config) -> jnp.ndarray:
    p = world.params
    gradient = p.ambient_amp * (p.ambient_dir @ world.agent_pos) / cfg.arena_half_width
    warmth = p.cg_warmth * jnp.exp(-jnp.linalg.norm(world.cg_pos - world.agent_pos) / cfg.warmth_scale)
    return jnp.clip(p.ambient_base + gradient + warmth, 0.0, 1.0)


def _proximity(world: World, cfg: Config) -> jnp.ndarray:
    return jnp.exp(-jnp.linalg.norm(world.cg_pos - world.agent_pos) / cfg.proximity_scale)


def _wall_clamp(pos: jnp.ndarray, vel: jnp.ndarray, radius: jnp.ndarray, cfg: Config):
    """Keep disks inside the arena; zero the outward velocity of any that hit a wall.

    Returns (pos, vel, impact_speed) where impact_speed is the summed outward speed removed.
    """
    lim = cfg.arena_half_width - radius[..., None]
    hit = jnp.abs(pos) > lim
    outward = hit & (pos * vel > 0)
    impact = jnp.sum(jnp.where(outward, jnp.abs(vel), 0.0), axis=-1)
    return jnp.clip(pos, -lim, lim), jnp.where(outward, 0.0, vel), impact


def _safe_unit(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unit vector and norm along the last axis; zero vector where the norm is zero."""
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.where(n > 0, n, 1.0), n[..., 0]


def step_world(world: World, action: jnp.ndarray, key: jax.Array, cfg: Config) -> World:
    """Advance the sandbox by one timestep of `cfg.dt`.

    Args:
        world: current state.
        action: (act_dim,) motor vector in [-1, 1].
        key: PRNG key for this step (caregiver wander noise). Split by the caller
            at the top of its loop (SPEC 12).
        cfg: global config.
    Returns:
        The next `World`, with body-contact signals for the homeostat filled in.
    Invariants: I9, I10.
    """
    dt = cfg.dt
    p = world.params
    thrust, turn, _, manipulate, vocalize = effectors(action)

    # --- organism body: thrust along heading, linear drag ---
    heading = world.agent_heading + turn * p.max_turn * dt
    fwd = jnp.array([jnp.cos(heading), jnp.sin(heading)])
    acc = thrust * p.max_thrust * fwd / p.agent_mass - p.friction * world.agent_vel
    a_vel = world.agent_vel + acc * dt
    a_pos = world.agent_pos + a_vel * dt

    # --- grasp: manipulating pulls every object within reach toward the mouth ---
    to_obj = world.obj_pos - a_pos
    gap = jnp.linalg.norm(to_obj, axis=-1) - p.obj_radius - p.agent_radius
    in_reach = (gap < cfg.reach)[:, None]
    mouth = a_pos + fwd * (p.agent_radius + p.obj_radius)[:, None]
    pull = cfg.grasp_k * manipulate * (mouth - world.obj_pos) * in_reach
    o_vel = world.obj_vel + (pull - p.friction * world.obj_vel) * dt
    o_pos = world.obj_pos + o_vel * dt

    # --- caregiver: scripted, kinematic; keeps a preferred distance, answers vocalization ---
    noise = jax.random.normal(key, (2,))
    wvel = world.cg_wvel - cfg.cg_wander_theta * world.cg_wvel * dt + p.cg_wander * jnp.sqrt(dt) * noise
    u_cg, d_cg = _safe_unit(a_pos - world.cg_pos)
    cg_vel = (p.cg_gain * (d_cg - p.cg_pref_dist) + p.cg_voc_response * vocalize) * u_cg + wvel
    cg_unit, cg_speed = _safe_unit(cg_vel)
    cg_vel = cg_unit * jnp.minimum(cg_speed, p.cg_max_speed)
    cg_pos = world.cg_pos + cg_vel * dt

    # --- organism vs objects: push apart by inverse mass, absorb approach speed (inelastic) ---
    n, d = _safe_unit(o_pos - a_pos)  # (K, 2), (K,) normals from organism to object
    overlap = jnp.maximum(p.agent_radius + p.obj_radius - d, 0.0)
    touching = overlap > 0
    inv_a, inv_o = 1.0 / p.agent_mass, 1.0 / p.obj_mass
    share_a = inv_a / (inv_a + inv_o)
    a_pos = a_pos - jnp.sum(n * (overlap * share_a)[:, None], axis=0)
    o_pos = o_pos + n * (overlap * (1.0 - share_a))[:, None]
    approach = jnp.where(touching, jnp.maximum(jnp.sum((a_vel - o_vel) * n, axis=-1), 0.0), 0.0)
    a_vel = a_vel - jnp.sum(n * (approach * share_a)[:, None], axis=0)
    o_vel = o_vel + n * (approach * (1.0 - share_a))[:, None]
    obj_impact = jnp.sum(approach)

    # --- object vs object: positional separation only ---
    diff = o_pos[:, None, :] - o_pos[None, :, :]
    nn, dd = _safe_unit(diff)
    ov = jnp.maximum(p.obj_radius[:, None] + p.obj_radius[None, :] - dd, 0.0)
    ov = ov * (1.0 - jnp.eye(cfg.n_objects))
    o_pos = o_pos + 0.5 * jnp.sum(nn * ov[..., None], axis=1)

    # --- organism vs caregiver: the caregiver is kinematic, so only the organism moves ---
    n_c, d_c = _safe_unit(a_pos - cg_pos)
    a_pos = a_pos + n_c * jnp.maximum(p.agent_radius + cfg.cg_radius - d_c, 0.0)

    # --- walls ---
    a_pos, a_vel, wall_impact = _wall_clamp(a_pos, a_vel, p.agent_radius, cfg)
    o_pos, o_vel, _ = _wall_clamp(o_pos, o_vel, p.obj_radius, cfg)
    cg_pos, _, _ = _wall_clamp(cg_pos, cg_vel, jnp.asarray(cfg.cg_radius), cfg)
    impact = wall_impact + obj_impact

    # --- hazard: damage scales with how deeply the body overlaps it ---
    d_h = jnp.linalg.norm(world.hazard_pos - a_pos)
    in_hazard = jnp.clip((p.hazard_radius + p.agent_radius - d_h) / p.agent_radius, 0.0, 1.0)
    # Only real collisions hurt: pressing against a wall re-clamps a tiny outward speed every step.
    hurt = jnp.maximum(impact - cfg.impact_threshold, 0.0)
    damage = cfg.hazard_damage_rate * in_hazard + cfg.impact_damage_gain * hurt / dt

    # --- food: manipulating the nearest stocked patch within reach ingests from it ---
    d_f = jnp.linalg.norm(world.food_pos - a_pos, axis=-1) - p.food_radius - p.agent_radius
    nearest = jax.nn.one_hot(jnp.argmin(d_f), cfg.n_food)
    eat = nearest * (d_f < cfg.reach) * (world.food_amount > 0) * cfg.ingest_rate * manipulate
    food = jnp.clip(world.food_amount - eat * dt + p.food_regrow * dt, 0.0, cfg.food_capacity)

    new = world.replace(
        agent_pos=a_pos,
        agent_vel=a_vel,
        agent_heading=jnp.arctan2(fwd[1], fwd[0]),  # wrap to (-pi, pi]
        obj_pos=o_pos,
        obj_vel=o_vel,
        food_amount=food,
        cg_pos=cg_pos,
        cg_wvel=wvel,
        last_action=jnp.clip(action, -1.0, 1.0),
        ingest=jnp.sum(eat),
        damage=damage,
        impact=impact,
    )
    return new.replace(ambient=_ambient_at(new, cfg), proximity=_proximity(new, cfg))


ODOR_SENSORS = 2  # left and right nostril


def obs_dim(cfg: Config) -> int:
    """Length of the observation vector returned by `sense`."""
    # per ray: depth + RGB; proprioception: body-frame velocity, last action, impact; smell; interoception
    return 4 * cfg.n_rays + 2 + cfg.act_dim + 1 + ODOR_SENSORS + cfg.n_drives


def odor_slice(cfg: Config) -> slice:
    """Where the two odor readings sit in the observation vector."""
    start = 4 * cfg.n_rays + 2 + cfg.act_dim + 1
    return slice(start, start + ODOR_SENSORS)


def odor(world: World, cfg: Config) -> jnp.ndarray:
    """Food odor at the two nostrils, shape (2,): Σ stock · exp(−distance / odor_scale).

    A raw chemical intensity: it says nothing about what or where the food is,
    but its left−right difference points toward it, even out of sight.
    """
    p = world.params
    side = jnp.array([1.0, -1.0]) * cfg.nostril_angle
    angles = world.agent_heading + side
    nostrils = world.agent_pos + p.agent_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1)  # (2, 2)
    d = jnp.linalg.norm(nostrils[:, None, :] - world.food_pos[None], axis=-1)  # (2, F)
    stock = world.food_amount / cfg.food_capacity
    return jnp.sum(stock * jnp.exp(-d / cfg.odor_scale), axis=-1)


def sense(world: World, drives: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Egocentric, low-level observation: rays + proprioception + interoception.

    Each ray reports `1 - distance/range` (0 = nothing in range) and the RGB
    colour of the first surface it hits. No labels or coordinates are exposed.

    Args:
        world: current state.
        drives: (n_drives,) interoceptive vector.
        cfg: global config.
    Returns:
        (obs_dim(cfg),) float vector.
    Invariants: I7 (interoception comes from body state), I10.
    """
    p = world.params
    angles = world.agent_heading + jnp.linspace(-cfg.ray_fov / 2, cfg.ray_fov / 2, cfg.n_rays)
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (R, 2)
    o = world.agent_pos

    food_tint = cfg.food_dim_floor + (1.0 - cfg.food_dim_floor) * world.food_amount / cfg.food_capacity
    centers = jnp.concatenate([world.obj_pos, world.food_pos, world.cg_pos[None], world.hazard_pos[None]])
    radii = jnp.concatenate(
        [p.obj_radius, p.food_radius, jnp.array([cfg.cg_radius]), p.hazard_radius[None]]
    )
    colors = jnp.concatenate(
        [
            jnp.broadcast_to(p.colors[OBJECT], (cfg.n_objects, 3)),
            p.colors[FOOD] * food_tint[:, None],
            p.colors[CAREGIVER][None],
            p.colors[HAZARD][None],
            p.colors[WALL][None],
        ]
    )

    # Ray–disk intersection for every (ray, disk) pair; a ray starting inside a disk hits at 0.
    oc = centers - o  # (C, 2)
    b = dirs @ oc.T  # (R, C) projection of centre onto ray
    h2 = radii**2 - (jnp.sum(oc**2, axis=-1) - b**2)
    h = jnp.sqrt(jnp.maximum(h2, 0.0))
    near, far = b - h, b + h
    t_disk = jnp.where(h2 >= 0, jnp.where(near > 0, near, jnp.where(far > 0, 0.0, jnp.inf)), jnp.inf)

    # Ray–wall: distance to the arena boundary along each ray (always hit from inside).
    wall = jnp.where(dirs > 0, cfg.arena_half_width, -cfg.arena_half_width)
    safe = jnp.where(dirs != 0, dirs, 1.0)
    t_wall = jnp.min(jnp.where(dirs != 0, (wall - o) / safe, jnp.inf), axis=-1)  # (R,)

    t_all = jnp.concatenate([t_disk, t_wall[:, None]], axis=-1)  # (R, C + 1)
    first = jnp.argmin(t_all, axis=-1)
    t = jnp.min(t_all, axis=-1)
    visible = t < cfg.ray_range
    depth = jnp.where(visible, 1.0 - t / cfg.ray_range, 0.0)
    rgb = colors[first] * visible[:, None]

    c, s = jnp.cos(world.agent_heading), jnp.sin(world.agent_heading)
    body_vel = jnp.array([c * world.agent_vel[0] + s * world.agent_vel[1], -s * world.agent_vel[0] + c * world.agent_vel[1]])
    return jnp.concatenate(
        [depth, rgb.reshape(-1), body_vel, world.last_action, world.impact[None], odor(world, cfg), drives]
    )
