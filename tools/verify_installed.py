"""Release gate for an installed wheel; never import the source checkout."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


def check(*, native: bool, fallback: bool) -> dict[str, object]:
    checkout = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("reality_stone")
    assert spec is not None and spec.origin, "reality_stone is not installed"
    installed = Path(spec.origin).resolve().parent
    assert not installed.is_relative_to(checkout), f"checkout import: {installed}"
    assert "ce-agi-runtime/reality_stone" not in installed.as_posix(), installed
    temporary = tempfile.TemporaryDirectory(prefix="reality-stone-fallback-") if fallback else None
    try:
        if temporary:
            shutil.copytree(installed, Path(temporary.name) / "reality_stone",
                            ignore=shutil.ignore_patterns("*.pyd", "*.so", "*.dll", "__pycache__"))
            sys.path.insert(0, temporary.name)
        import numpy as np
        import torch
        import reality_stone as rs
        import reality_stone.clarus as clarus
        from reality_stone._fallback import mobius_add_torch
        from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode

        assert rs.__version__ == "0.3.0", rs.__version__
        assert clarus.__version__ == rs.__version__
        assert rs._has_rust_ext == native
        assert clarus.has_native_kernels() == native
        if native:
            from reality_stone import _rust as geometry
            from reality_stone.clarus import _rust as runtime_native
            for extension in (geometry, runtime_native):
                assert Path(extension.__file__).resolve().is_relative_to(installed)
                assert extension.__file__.endswith((".pyd", ".so"))

        x = torch.tensor([[0.10, -0.05], [0.03, 0.07]], requires_grad=True)
        y = torch.tensor([[0.02, 0.04], [-0.01, 0.03]], requires_grad=True)
        actual = rs.poincare_add(x, y, c=0.7)
        actual.square().sum().backward()
        xr, yr = x.detach().requires_grad_(), y.detach().requires_grad_()
        reference = mobius_add_torch(xr, yr, 0.7)
        reference.square().sum().backward()
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(x.grad, xr.grad, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(y.grad, yr.grad, atol=2e-4, rtol=2e-3)

        weights = torch.tensor(np.arange(64).reshape(8, 8) / 2000, dtype=torch.float32)
        weights = (weights + weights.T) / 2
        weights.fill_diagonal_(0)
        backends = ["torch", "rust"] if native else ["torch", "auto"]
        states = []
        snapshots = []
        for backend in backends:
            torch.manual_seed(7)
            runtime = BrainRuntime(weights.clone(), config=BrainRuntimeConfig(dim=8, noise_sigma=0.0, axon_delay=False),
                                   backend=backend, device="cpu")
            energies = []
            for _ in range(3):
                step = runtime.step(external_input=torch.linspace(0, 0.2, 8), force_mode=RuntimeMode.WAKE)
                assert step.mode == RuntimeMode.WAKE
                assert np.isfinite(step.energy)
                energies.append(float(step.energy))
            states.append(energies)
            snapshots.append(runtime.snapshot())
        np.testing.assert_allclose(states[0], states[1], rtol=2e-3, atol=2e-4)
        for name in ("activation", "refractory", "memory_trace", "adaptation", "stp_u", "stp_x", "bitfield"):
            torch.testing.assert_close(getattr(snapshots[0], name), getattr(snapshots[1], name), atol=2e-4, rtol=2e-3)
        return {"status": "PASS", "version": rs.__version__, "native": native,
                "fallback": fallback, "package": str(Path(rs.__file__).resolve()),
                "geometry_gradient": "PASS", "runtime_backends": backends, "runtime_energies": states}
    finally:
        if temporary:
            sys.path.remove(temporary.name)
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--native", action="store_true")
    group.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(native=args.native, fallback=args.fallback), sort_keys=True))
