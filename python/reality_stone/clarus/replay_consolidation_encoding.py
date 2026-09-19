from __future__ import annotations

import hashlib
import math

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1e-12:
        return np.zeros_like(value)
    return value / norm


def _stable_seed(seed: int, namespace: str, token: str) -> int:
    payload = f"{int(seed)}|{namespace}|{token}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class StableBipolarEncoder:
    """Deterministic distributed codes with no learned key/value pairing."""

    def __init__(self, *, seed: int, dimension: int) -> None:
        self.seed = int(seed)
        self.dimension = int(dimension)
        if self.dimension < 8:
            raise ValueError("dimension must be at least 8")

    def code(self, namespace: str, token: str) -> np.ndarray:
        rng = np.random.default_rng(_stable_seed(self.seed, namespace, str(token)))
        vector = rng.choice(np.array([-1.0, 1.0]), size=self.dimension)
        return vector / math.sqrt(self.dimension)

    def key(self, subject: str, relation: str) -> np.ndarray:
        bound = self.code("subject", subject) * self.code("relation", relation)
        return normalize(bound)

    def value(self, value: str) -> np.ndarray:
        return self.code("value", value)

    @staticmethod
    def corrupt(vector: np.ndarray, *, flip_rate: float, rng: np.random.Generator) -> np.ndarray:
        rate = float(flip_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("flip_rate must lie in [0, 1]")
        result = np.asarray(vector, dtype=np.float64).copy()
        if rate > 0.0:
            result[rng.random(result.size) < rate] *= -1.0
        return normalize(result)
