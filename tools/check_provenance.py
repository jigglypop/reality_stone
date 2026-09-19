"""Check that extraction did not rewrite algorithms or frozen research files."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "ce-source-manifest.json").read_text(encoding="utf-8"))
packaging_changes = {
    ".gitignore", "Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md",
    "python/reality_stone/__init__.py", "python/reality_stone/clarus/__init__.py",
    "python/reality_stone/clarus/core/Cargo.toml",
    "python/reality_stone/clarus/core/Cargo.lock",
}
# A release-gate gradient check found a missing numerator term in the old VJP.
# The original bytes stay in ce-source-manifest.json and the preserved checkout.
release_fixes = {"src/ops/mobius.rs"}
failures = []
checked = 0
for relative, expected in manifest["source_files"].items():
    if relative in packaging_changes | release_fixes:
        continue
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(relative)
    checked += 1
if failures:
    raise SystemExit("Extraction changed source bytes: " + ", ".join(failures))
print(json.dumps({"status": "PASS", "unchanged_files": checked,
                  "source_commit": manifest["source_commit"]}))
