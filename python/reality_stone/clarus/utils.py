"""Shared utilities used across the clarus package.

Consolidates safe_print, normalize_vector, and resolve_device
which were previously duplicated in engine.py, runtime.py, and sleep.py.
"""

from __future__ import annotations

import sys

import torch

from .constants import NORM_EPS


def safe_print(text: object) -> None:
    """Print with UTF-8 fallback for Windows consoles."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        data = (str(text) + "\n").encode("utf-8", errors="replace")
        sys.stdout.buffer.write(data)
        sys.stdout.flush()


def normalize_vector(x: torch.Tensor) -> torch.Tensor:
    """Detach, cast to float, and L2-normalize. Returns zeros on degenerate input."""
    x = x.detach().float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm.item() < NORM_EPS:
        return torch.zeros_like(x)
    return x / norm


def resolve_device(name: str) -> torch.device:
    """Resolve a device name to a torch.device with validation."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
