"""CUDA / CPU device auto-detection and selection."""

from __future__ import annotations

import torch


def auto_device(preference: str = "auto") -> torch.device:
    """Return the best available device.

    Args:
        preference: "cuda", "cpu", or "auto" (default).
            "auto" picks CUDA if available, else CPU.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


def device_summary(device: torch.device) -> str:
    """One-line summary of the compute device."""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_mem / 1e9
        return f"CUDA:{idx} {name} ({mem_gb:.1f} GB)"
    return "CPU"
