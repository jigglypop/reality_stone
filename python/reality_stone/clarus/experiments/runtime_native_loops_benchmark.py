"""CLI for deterministic development or confirmation native-loop runs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .runtime_native_loops import run_route_b_seed_range, run_seed_range

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--route-b", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = range(98101, 98133) if args.confirmation else range(97101, 97109)
    results = run_route_b_seed_range(seeds) if args.route_b else run_seed_range(seeds)
    payload = json.dumps(results, sort_keys=True)
    mode = (
        ("route_b_confirmation" if args.confirmation else "route_b_development")
        if args.route_b else ("confirmation" if args.confirmation else "development")
    )
    report = json.dumps({"mode": mode, "results": results, "sha256": hashlib.sha256(payload.encode()).hexdigest()}, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)

if __name__ == "__main__": main()
