# CE-BRAIN extraction

The source snapshot is CE-BRAIN commit
`fffd356ee4f1f7bf5079f3379cd06e4dc56a444c` from
https://github.com/jigglypop/CE-BRAIN.
The independent repository retains its previous history, beginning this change
from `c538bf0ffa0ab96dbe04701a81200708828fe4af`.

`ce-source-manifest.json` records the SHA-256 of every imported source file
before packaging changes. The original repository, including its source,
tests, receipts, and relative layout, was copied to
`C:\dev\ce\ce-agi-runtime-repro-fffd356`; all 2,590 tracked working-tree files
were compared byte-for-byte. External data and the read-only `ce-runs` store
remain external. This archive is a preserved software environment, not a claim
that every historical experiment can run without its original inputs.

The initial extraction preserves `reality_stone.*` and
`reality_stone.clarus.*`. Version and distribution metadata change; algorithms
and hash-locked research files remain unchanged except for the release fix below.
Frozen research tests should
run in the preserved CE-BRAIN environment. New package integration tests run
against an installed wheel, outside either source tree.

Consumers should depend on `reality_stone==0.3.0`, remove checkout-specific
`PYTHONPATH` injection, and resolve active Python modules by import rather than
the old `reality_stone/python/reality_stone` filesystem path. Historic source
hash receipts continue to refer to the original snapshot; do not rewrite their
hashes to make a relocated execution appear identical.

## Release correctness fix

Installed native-wheel testing exposed a missing numerator contribution to the
second-input gradient in `src/ops/mobius.rs::mobius_add_vjp`. Version 0.3.0 adds
the derivative of `c * |y|^2 * x`, and differentiates the same denominator clamp
as the forward operation. The source manifest retains the original hash. This
file is not a locked research source; no frozen receipt or research algorithm
has been rewritten. Native and Python gradients are checked against the Torch
formula during the wheel release gate.
