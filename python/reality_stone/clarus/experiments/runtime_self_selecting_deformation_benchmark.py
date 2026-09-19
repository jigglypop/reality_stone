"""Discovery-only CLI for M4-R.  Validation and confirmation are sealed."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from reality_stone.clarus.experiments.runtime_self_selecting_deformation import (
    DISCOVERY_SEEDS, SelfSelectingDeformationConfig, self_selecting_deformation,
)

def main() -> None:
    # Discovery seeds are process-independent.  One CPU thread per worker
    # avoids oversubscribing the host when the runner is sharded externally.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seeds", default="97401:97408"); parser.add_argument("--merge", type=Path, nargs="*")
    args = parser.parse_args()
    if args.merge:
        rows = []
        for path in args.merge:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["mode"] != "discovery" or payload["status"] not in {"COMPLETE", "RUNNING"}:
                raise SystemExit(f"incomplete/non-discovery shard: {path}")
            rows.extend(payload["results"])
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != list(DISCOVERY_SEEDS):
            raise SystemExit("merge requires exactly discovery seeds 97401..97408")
        merged = {"mode":"discovery","status":"COMPLETE","seed_range":[97401,97408],"validation_opened":False,"confirmation_opened":False,
            "source_sha256":hashlib.sha256(Path(__file__).with_name("runtime_self_selecting_deformation.py").read_bytes()).hexdigest(),"results":rows}
        merged["result_sha256"] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        args.output.write_text(json.dumps(merged,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(json.dumps({"merged_seeds":len(rows)},sort_keys=True)); return
    if ":" in args.seeds:
        start, end = (int(value) for value in args.seeds.split(":", 1)); seeds = list(range(start, end + 1))
    else:
        seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or any(seed not in DISCOVERY_SEEDS for seed in seeds):
        raise SystemExit("only discovery seeds 97401..97408 are authorized")
    rows = []
    def write(status: str) -> None:
        payload = {"mode":"discovery","status":status,"seed_range":[min(seeds),max(seeds)],"validation_opened":False,"confirmation_opened":False,
          "source_sha256":hashlib.sha256(Path(__file__).with_name("runtime_self_selecting_deformation.py").read_bytes()).hexdigest(),"results":rows}
        payload["result_sha256"] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for seed in seeds:
        rows.append(self_selecting_deformation(seed, SelfSelectingDeformationConfig(seed=seed)))
        write("RUNNING")
    write("COMPLETE")
    print(json.dumps({"seeds":len(rows),"status_counts":{x:sum(r["status"]==x for r in rows) for x in ("GO","STOP")}},sort_keys=True))
if __name__ == "__main__": main()
