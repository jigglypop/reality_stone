# Reality Stone

Reality Stone provides PyTorch geometry operators and the experimental Clarus
runtime. Release 0.3.0 extracts the implementation used by CE-BRAIN into one
installable package. Both Rust extensions are built into each CPU wheel:
`reality_stone._rust` and `reality_stone.clarus._rust`.

## Install

```sh
python -m pip install reality_stone==0.3.0
# Optional language-model integration:
python -m pip install "reality_stone[llm]==0.3.0"
```

CPython 3.10–3.12 is supported. Release wheels target Windows x86-64, Linux
x86-64 (glibc 2.28+), and macOS arm64. NumPy and PyTorch are required. Installing
a wheel does not require Rust. Other platforms require a Rust toolchain and a
native compiler to build the source distribution.

## Geometry and gradients

```python
import torch
import reality_stone as rs

x = torch.tensor([[0.1, 0.2]], requires_grad=True)
y = rs.poincare_add(x, torch.zeros_like(x), c=1.0)
y.square().sum().backward()
assert torch.isfinite(x.grad).all()
print(rs.__version__, rs._has_rust_ext, rs._has_cuda)
```

## Clarus runtime

```python
import torch
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode

weights = torch.zeros(8, 8)
runtime = BrainRuntime(
    weights, config=BrainRuntimeConfig(dim=8, noise_sigma=0.0),
    backend="torch", device="cpu",
)
step = runtime.step(external_input=torch.ones(8) * 0.1, force_mode=RuntimeMode.WAKE)
print(step.mode, step.energy)
```

Use `backend="rust"` to require the native Clarus kernel, or `"auto"` to select
an available backend. Native runtime configuration requires `axon_delay=False`;
delayed axons, neuronwise bit thresholds, and local competition use Torch.
`reality_stone.clarus.has_native_kernels()` reports whether
the Clarus extension loaded. `rs._has_rust_ext` describes the separate geometry
extension; successful import alone does not prove that both extensions loaded.

## Support and stability

* Geometry: public PyTorch layers, transformations, losses, and optimizers.
* Clarus runtime: experimental simulation APIs; no claim of biological or AGI
  validation follows from installation or the software checks.
* Research modules (`clarus.experiments`, `verified_*`, `quantitative_*`, probes,
  and frozen bridges): retained for import compatibility. Their original
  datasets, contracts, receipts, and checkout layout are external prerequisites.
  The wheel alone is not a historical research reproduction environment.
* Native modules are private implementation details. Low-level Rust calls have
  narrower dtype/shape contracts than PyTorch operations.
* Without native extensions, supported geometry operations and the Torch
  runtime use the existing Python fallback. Transformer-to-RSULF conversion
  still requires the geometry extension. Source builds require Rust; native
  build errors are not silently converted into incomplete release wheels.
* CUDA kernels are experimental and excluded from the CPU release wheels.
  `_has_cuda == False` does not prevent ordinary PyTorch CUDA computations.
  Native CUDA support is not claimed by this release.

Optional groups are `llm`, `science`, `vision`, `quantum`, `neuro`, and `dev`.
Model and dataset downloads occur only in the corresponding higher-level APIs;
they are not part of the installation checks.

## Build and validate

```sh
python -m pip install build
python -m build
python -m pip install --force-reinstall --no-deps dist/reality_stone-0.3.0-*.whl
# Run from outside this checkout:
python /path/to/reality_stone/tools/verify_installed.py --native
python /path/to/reality_stone/tools/verify_installed.py --fallback
```

The installed-package check rejects checkout imports, checks both extension
origins, compares a geometry gradient with the Torch formula, and executes
Clarus with both native and Torch backends. Fallback checking copies only the
installed Python package into a temporary directory without native libraries.

The existing repository documents describe historical research and may refer
to earlier APIs. See [migration and provenance](MIGRATION.md) for this release.
Licensed under [MIT](LICENSE).
