"""Evolution: rank-normalized evolution strategy over the flat genome (SPEC 7.14).

Each generation samples `pop_size` mutants around the mean, scores them by
developmental fitness, and moves the mean along the rank-weighted noise.
The SPEC calls this CMA-ES, but its update is an isotropic (separable)
evolution strategy; a full covariance over ~2.8·10^5 genes is infeasible,
so the SPEC's code is what is implemented (DECISIONS D50).

All randomness derives from one root key and the generation number, so a
run restored from a checkpoint continues bit-identically.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from creature.config import Config
from creature.outer.fitness import fitness_components
from creature.outer.genome import Genome, init_genome

FitnessFn = Callable[[jnp.ndarray, jax.Array, Config], jnp.ndarray]


@struct.dataclass
class EvolveState:
    mean: jnp.ndarray  # (genome_dim,) flat genome
    sigma: jnp.ndarray  # () mutation scale
    gen: jnp.ndarray  # () int32 generations completed
    parent_fit: jnp.ndarray  # () fitness of the mean genome at the last generation
    root: jnp.ndarray  # raw key data of the run's root key


def rank_normalize(fits: jnp.ndarray) -> jnp.ndarray:
    """Centered ranks in [−1, 1] with zero mean: robust to fitness outliers and rejections."""
    n = fits.shape[0]
    ranks = jnp.argsort(jnp.argsort(fits))
    return 2.0 * ranks / (n - 1) - 1.0


def adapt_sigma(sigma: jnp.ndarray, fits: jnp.ndarray, parent_fit: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """1/5th success rule: grow sigma when more than the target share of mutants beats their parent
    (the current mean genome, evaluated in the same worlds), shrink it otherwise."""
    success = jnp.mean(fits > parent_fit)
    sigma = sigma * jnp.exp((success - cfg.es_success_target) / cfg.es_sigma_damping)
    return jnp.clip(sigma, *cfg.es_sigma_bounds)


def genome_fitness(flat: jnp.ndarray, key: jax.Array, cfg: Config) -> jnp.ndarray:
    """Developmental fitness of a flat genome vector."""
    return fitness_components(Genome.from_flat(flat, cfg), key, cfg)["fitness"]


def init_state(rng: jax.Array, cfg: Config) -> EvolveState:
    k_init, k_root = jax.random.split(rng)
    return EvolveState(
        mean=init_genome(k_init, cfg).flat(),
        sigma=jnp.asarray(cfg.es_sigma_init),
        gen=jnp.zeros((), jnp.int32),
        parent_fit=jnp.asarray(-jnp.inf),
        root=jax.random.key_data(k_root),
    )


def generation(state: EvolveState, cfg: Config, fitness_fn: FitnessFn, eval_batch: int) -> tuple[EvolveState, jnp.ndarray]:
    """One generation. Returns the new state and the candidates' fitnesses."""
    key = jax.random.fold_in(jax.random.wrap_key_data(state.root), state.gen)
    k_eps, k_fit = jax.random.split(key)
    eps = jax.random.normal(k_eps, (cfg.pop_size, state.mean.shape[0]))
    cands = state.mean + state.sigma * eps
    # Common random numbers: every candidate, and the parent, lives in the same worlds this generation.
    everyone = jnp.concatenate([state.mean[None], cands])
    all_fits = jax.lax.map(lambda c: fitness_fn(c, k_fit, cfg), everyone, batch_size=eval_batch)
    parent_fit, fits = all_fits[0], all_fits[1:]
    r = rank_normalize(fits)
    # Step ∝ sigma (CMA-style), not ∝ 1/sigma as in the SPEC sketch: with a shrinking
    # sigma the 1/sigma form amplifies noise until it diverges (DECISIONS D50).
    mean = state.mean + cfg.es_lr * state.sigma * (eps.T @ r) / cfg.pop_size
    sigma = adapt_sigma(state.sigma, fits, parent_fit, cfg)
    return EvolveState(mean=mean, sigma=sigma, gen=state.gen + 1, parent_fit=parent_fit, root=state.root), fits


def save_state(state: EvolveState, path: str | Path) -> None:
    np.savez(path, **{f: np.asarray(getattr(state, f)) for f in state.__dataclass_fields__})


def load_state(path: str | Path) -> EvolveState:
    with np.load(path) as d:
        return EvolveState(**{f: jnp.asarray(d[f]) for f in d.files})


def evolve(
    cfg: Config,
    rng: jax.Array,
    generations: int | None = None,
    fitness_fn: FitnessFn = genome_fitness,
    checkpoint: str | Path | None = None,
    log: Callable[[int, jnp.ndarray, EvolveState], None] | None = None,
    eval_batch: int = 8,
) -> EvolveState:
    """Run evolution for `generations` (default `cfg.generations`).

    If `checkpoint` exists the run resumes from it; the state is saved there
    after every generation. `log(gen, fits, state)` is called each generation;
    `state.parent_fit` is the fitness of the generation's mean genome.
    Returns the final state; `Genome.from_flat(state.mean)` is the endowment.
    """
    total = cfg.generations if generations is None else generations
    state = load_state(checkpoint) if checkpoint is not None and Path(checkpoint).exists() else init_state(rng, cfg)
    step = jax.jit(lambda s: generation(s, cfg, fitness_fn, eval_batch))
    while int(state.gen) < total:
        state, fits = step(state)
        if log is not None:
            log(int(state.gen), fits, state)
        if checkpoint is not None:
            save_state(state, checkpoint)
    return state
