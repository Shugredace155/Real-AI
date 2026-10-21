"""Developmental fitness (SPEC 7.13, 9). The most important function in the project.

A genome is scored by the lives it produces: it is born into
`n_eval_worlds` randomized worlds, lives `dev_steps` steps in each (with sleep
on schedule), has its body perturbed halfway through, and is judged on how
it develops, not only on how it ends up:

    survival     fraction of life alive
    learn_speed  relative fall of the PCN's error on a fixed probe set of views of the
                 world, first → last quarter (weighted 3×: anti-statue). Measured on
                 views the organism must go and see, so stillness cannot score (D58)
    competence   last-quarter advantage: health minus the health of a resting twin
                 in an identical world (life.Twin), i.e. what behavior adds
    robustness   last-quarter health relative to the quarter before the body perturbation
    curiosity    mean learning progress (intrinsic reward; not scaled by any gene)
    hack         fraction of life with no plasticity + trivially low homeostatic error

Degenerate genomes are rejected outright (SPEC 9.2): statues (no
plasticity on > `statue_fraction` of life) and genomes whose setpoints are
already met at birth.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from creature import life
from creature import organism as O
from creature.affect import core as affect
from creature.body import sandbox as sb
from creature.config import Config
from creature.outer.genome import Genome, express

EPS = 1e-6


def dev_config(cfg: Config) -> Config:
    """Developmental evaluation runs many lives at once; the episodic buffer is shrunk to fit."""
    return dataclasses.replace(cfg, episodic_capacity=cfg.dev_episodic_capacity)


def perturb(world: sb.World, key: jax.Array, cfg: Config) -> sb.World:
    """Change the body: mass, thrust and turning each scaled by U(perturb_scale)."""
    s = jax.random.uniform(key, (3,), minval=cfg.perturb_scale[0], maxval=cfg.perturb_scale[1])
    p = world.params
    return world.replace(params=p.replace(agent_mass=p.agent_mass * s[0], max_thrust=p.max_thrust * s[1], max_turn=p.max_turn * s[2]))


def _quarters(x: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    q = x.shape[0] // 4
    return x[:q], x[q : 2 * q], x[2 * q : 3 * q], x[3 * q :]


def life_components(stats: dict, g: O.Endowment, cfg: Config) -> dict:
    """All fitness terms from one life's binned statistics."""
    first, _, _, last = _quarters(stats["probe_err"])
    _, _, h_before, h_last = _quarters(stats["health"])
    adv1, _, _, adv_last = _quarters(stats["advantage"])
    survival = stats["alive"].mean()
    learn_speed = (first.mean() - last.mean()) / (first.mean() + EPS)  # RELATIVE
    competence = adv_last.mean()
    robustness = jnp.clip(h_last.mean() / (h_before.mean() + EPS), 0.0, 1.0)
    curiosity = stats["intrinsic"].mean()
    statue = (stats["plasticity"] < cfg.statue_plasticity).mean()
    birth_err = affect.error_magnitude(jnp.asarray(cfg.drive_init), g)
    trivial = (stats["err_mag"].mean() < cfg.trivial_err).astype(jnp.float32)
    hack = statue + trivial
    total = (
        cfg.w_survival * survival
        + cfg.w_learn * learn_speed
        + cfg.w_competence * competence
        + cfg.w_robustness * robustness
        + cfg.w_curiosity * curiosity
        - cfg.w_hack * hack
    )
    rejected = (statue > cfg.statue_fraction) | (birth_err < cfg.trivial_err)
    return dict(
        fitness=jnp.where(rejected, cfg.reject_fitness, total),
        survival=survival,
        learn_speed=learn_speed,
        competence=competence,
        robustness=robustness,
        curiosity=curiosity,
        hack=hack,
        rejected=rejected.astype(jnp.float32),
        competence_birth=adv1.mean(),
        growth_ratio=growth_ratio(adv1.mean(), competence),
    )


def growth_ratio(competence_birth: jnp.ndarray, competence_end: jnp.ndarray) -> jnp.ndarray:
    """SPEC 9.3: share of final competence acquired during life. Clipped to [−1, 1];
    an organism with no positive final competence scores −1."""
    ratio = jnp.clip((competence_end - competence_birth) / jnp.maximum(competence_end, EPS), -1.0, 1.0)
    return jnp.where(competence_end > EPS, ratio, -1.0)


def fitness_components(genome: Genome, rng: jax.Array, cfg: Config) -> dict:
    """Mean of every fitness term over `n_eval_worlds` lives.

    Args:
        genome: the genome to evaluate.
        rng: PRNG key; worlds, lives and perturbations all derive from it.
        cfg: global config.
    Invariants: I8 (competence the individual could learn is not rewarded at
        birth: learn_speed and robustness dominate), I9, I10.
    """
    dcfg = dev_config(cfg)
    g = express(genome, dcfg)
    half = dcfg.dev_steps // 2

    def one_world(key):
        k_world, k_life, k_perturb, k_probe = jax.random.split(key, 4)
        env = sb.randomize_world(k_world, dcfg)
        probe = life.make_probe(env, k_probe, dcfg)
        agent = O.spawn(g, dcfg)
        bump = lambda o, w, t: (o, perturb(w, k_perturb, dcfg))
        twin = life.make_twin(env, dcfg)
        _, _, stats = life.live(agent, env, k_life, 0, dcfg.dev_steps, dcfg.fitness_bin_steps, dcfg, bump, (half,), twin, probe)
        return life_components(stats, g, dcfg)

    per_world = jax.vmap(one_world)(jax.random.split(rng, cfg.n_eval_worlds))
    return jax.tree.map(jnp.mean, per_world)


def developmental_fitness(genome: Genome, rng: jax.Array, cfg: Config) -> jnp.ndarray:
    """Scalar fitness (SPEC 7.13)."""
    return fitness_components(genome, rng, cfg)["fitness"]
