"""Episodic buffer: write-once, affect-tagged, salience-decayed (SPEC 7.8).

A ring buffer of traces. Each trace is written once, at the ring pointer, and
is only ever replaced when the ring wraps. Its salience is |valence| +
prediction error (scaled by the CORT-driven encoding knob) and decays with
age. Replay samples traces with probability softmax(effective salience /
temp), so a single intense event, replayed many times, carries the weight of
many mundane ones.

Age is stored as write time plus a shared clock, so `tick` is O(1) rather
than touching every slot each millisecond.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from creature.config import Config


class Episodic(struct.PyTreeNode):
    k: jnp.ndarray  # (N, dk) keys: context (latent state at the start of the trace)
    v: jnp.ndarray  # (N, dv) values: the observation sequence
    sal: jnp.ndarray  # (N,) salience at encoding = (|valence| + pred_err) · encoding
    t_write: jnp.ndarray  # (N,) clock time of the write, seconds
    now: jnp.ndarray  # () the buffer's clock, seconds
    ptr: jnp.ndarray  # () int32 next slot to write
    fill: jnp.ndarray  # () int32 number of slots ever written (saturates at N)

    @classmethod
    def create(cls, capacity: int, dk: int, dv: int) -> Episodic:
        """An empty buffer."""
        return cls(
            k=jnp.zeros((capacity, dk)),
            v=jnp.zeros((capacity, dv)),
            sal=jnp.zeros(capacity),
            t_write=jnp.zeros(capacity),
            now=jnp.zeros(()),
            ptr=jnp.zeros((), jnp.int32),
            fill=jnp.zeros((), jnp.int32),
        )

    @property
    def capacity(self) -> int:
        return self.sal.shape[-1]

    def write(
        self, key: jnp.ndarray, value: jnp.ndarray, valence: jnp.ndarray, pred_err: jnp.ndarray, encoding: jnp.ndarray = 1.0
    ) -> Episodic:
        """Write one trace at the ring pointer. Write-once: nothing else ever modifies a slot.

        Invariant I10 (pure; returns a new buffer).
        """
        i = self.ptr
        return self.replace(
            k=self.k.at[i].set(key),
            v=self.v.at[i].set(value),
            sal=self.sal.at[i].set((jnp.abs(valence) + pred_err) * encoding),
            t_write=self.t_write.at[i].set(self.now),
            ptr=(i + 1) % self.capacity,
            fill=jnp.minimum(self.fill + 1, self.capacity),
        )

    def age(self) -> jnp.ndarray:
        """(N,) seconds since each trace was written."""
        return self.now - self.t_write

    def effective_salience(self, cfg: Config) -> jnp.ndarray:
        """(N,) salience after age decay; −inf for never-written slots."""
        written = jnp.arange(self.capacity) < self.fill
        decayed = self.sal * jnp.exp(-self.age() * cfg.episodic_sal_decay)
        return jnp.where(written, decayed, -jnp.inf)

    def replay_batch(self, rng: jax.Array, k: int, cfg: Config) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample k traces, biased toward high salience. Returns (keys, values, indices). Invariant I9."""
        logits = self.effective_salience(cfg) / cfg.episodic_temp
        idx = jax.random.categorical(rng, logits, shape=(k,))
        return self.k[idx], self.v[idx], idx

    def tick(self, dt: float) -> Episodic:
        """Advance the clock; every trace ages, so old salience decays."""
        return self.replace(now=self.now + dt)


def save_episodic(ep: Episodic, path: str | Path) -> None:
    """Serialize to .npz so a crash is not a death (SPEC 7.8)."""
    np.savez(path, **{f.name: np.asarray(getattr(ep, f.name)) for f in ep.__dataclass_fields__.values()})


def load_episodic(path: str | Path) -> Episodic:
    with np.load(path) as data:
        return Episodic(**{name: jnp.asarray(data[name]) for name in data.files})
