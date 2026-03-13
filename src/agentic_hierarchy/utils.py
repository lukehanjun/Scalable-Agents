from __future__ import annotations

import hashlib
from statistics import mean


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def stable_float(value: str, lower: float = 0.0, upper: float = 1.0) -> float:
    ratio = stable_hash(value) / float(16**16 - 1)
    return lower + (upper - lower) * ratio


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0
