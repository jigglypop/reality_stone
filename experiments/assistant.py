from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lbigd.envs.combat import GridCombatEnv, DIR_MIN, DIR_MAX, RANGE_MIN


def run_balance(
    n_factions: int = 8,
    counts: str = "25,23,20,18,16,14,12,10",
    n_games: int = 200,
    board_size: int = 12,
    max_steps: int = 800,
    spawn_mode: str = "zones",
    n_rounds: int = 10,
    lr: float = 0.3,
    game_mode: str = "normal",
    start_mode: str = "rotate",
    obstacle_mode: str = "none",
    n_obstacles: int = 0,
) -> Dict[str, Any]:
    count_list = [int(x.strip()) for x in counts.split(",")]
    if len(count_list) != n_factions:
        count_list = count_list[:n_factions] if len(count_list) > n_factions else count_list + [10] * (n_factions - len(count_list))
    
    config = {
        "board_size": board_size,
        "n_factions": n_factions,
        "game_mode": game_mode,
        "max_steps": max_steps,
        "spawn_mode": spawn_mode,
        "start_mode": start_mode,
        "obstacle_mode": obstacle_mode,
        "n_obstacles": int(n_obstacles),
    }
    for f, c in enumerate(count_list):
        config[f"p{f}_units"] = c

    env = GridCombatEnv(config)
    result = env.simulate(n_games, n_rounds=int(n_rounds), lr=float(lr))

    print("=" * 50, flush=True)
    print("BALANCE REPORT", flush=True)
    print("=" * 50, flush=True)
    print(f"factions: {n_factions}", flush=True)
    print(f"counts: {count_list}", flush=True)
    print(f"patterns (dirs, range): {result['patterns']}", flush=True)
    print(f"wins: {result['wins']}", flush=True)
    print(f"rates: {result['rates']}", flush=True)
    print(f"smoothed: {result['smoothed']}", flush=True)
    
    return result


def run_uniform(
    n_factions: int = 8,
    counts: str = "25,23,20,18,16,14,12,10",
    n_games: int = 800,
    board_size: int = 12,
    max_steps: int = 800,
    spawn_mode: str = "global",
    case: str = "both",
    dir_count: int = DIR_MIN,
    max_range: int = RANGE_MIN,
    game_mode: str = "normal",
    start_mode: str = "rotate",
    obstacle_mode: str = "none",
    n_obstacles: int = 0,
) -> Dict[str, Any]:
    count_list = [int(x.strip()) for x in counts.split(",")]
    if len(count_list) != n_factions:
        count_list = count_list[:n_factions] if len(count_list) > n_factions else count_list + [10] * (n_factions - len(count_list))

    config = {
        "board_size": board_size,
        "n_factions": n_factions,
        "game_mode": game_mode,
        "max_steps": max_steps,
        "spawn_mode": spawn_mode,
        "start_mode": start_mode,
        "obstacle_mode": obstacle_mode,
        "n_obstacles": int(n_obstacles),
    }
    for f, c in enumerate(count_list):
        config[f"p{f}_units"] = c

    out: Dict[str, Any] = {"factions": n_factions, "counts": count_list}

    if case in ("min", "both"):
        env_min = GridCombatEnv(config)
        env_min.set_uniform_pattern(DIR_MIN, RANGE_MIN)
        res_min = env_min.simulate(n_games=n_games, n_rounds=0)
        out["all_min"] = {"dir": DIR_MIN, "range": RANGE_MIN, "wins": res_min["wins"], "rates": res_min["rates"]}

    if case in ("max", "both"):
        env_max = GridCombatEnv(config)
        env_max.set_uniform_pattern(DIR_MAX, board_size)
        res_max = env_max.simulate(n_games=n_games, n_rounds=0)
        out["all_max"] = {"dir": DIR_MAX, "range": board_size, "wins": res_max["wins"], "rates": res_max["rates"]}

    if case == "custom":
        env = GridCombatEnv(config)
        env.set_uniform_pattern(int(dir_count), int(max_range))
        res = env.simulate(n_games=n_games, n_rounds=0)
        out["custom"] = {"dir": int(dir_count), "range": int(max_range), "wins": res["wins"], "rates": res["rates"]}

    print("=" * 50, flush=True)
    print("UNIFORM REPORT", flush=True)
    print("=" * 50, flush=True)
    print(f"factions: {n_factions}", flush=True)
    print(f"counts: {count_list}", flush=True)
    if "all_min" in out:
        print(f"all_min: {out['all_min']}", flush=True)
    if "all_max" in out:
        print(f"all_max: {out['all_max']}", flush=True)
    if "custom" in out:
        print(f"custom: {out['custom']}", flush=True)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_bal = sub.add_parser("balance", aliases=["b"])
    p_bal.add_argument("--n-factions", type=int, default=8)
    p_bal.add_argument("--counts", type=str, default="25,23,20,18,16,14,12,10")
    p_bal.add_argument("--n-games", type=int, default=200)
    p_bal.add_argument("--board-size", type=int, default=12)
    p_bal.add_argument("--max-steps", type=int, default=800)
    p_bal.add_argument("--spawn-mode", type=str, default="zones")
    p_bal.add_argument("--n-rounds", type=int, default=10)
    p_bal.add_argument("--lr", type=float, default=0.3)
    p_bal.add_argument("--game-mode", type=str, default="normal")
    p_bal.add_argument("--start-mode", type=str, choices=["rotate", "random"], default="rotate")
    p_bal.add_argument("--obstacle-mode", type=str, choices=["none", "random", "zones"], default="none")
    p_bal.add_argument("--n-obstacles", type=int, default=0)

    p_uni = sub.add_parser("uniform", aliases=["u"])
    p_uni.add_argument("--n-factions", type=int, default=8)
    p_uni.add_argument("--counts", type=str, default="25,23,20,18,16,14,12,10")
    p_uni.add_argument("--n-games", type=int, default=800)
    p_uni.add_argument("--board-size", type=int, default=12)
    p_uni.add_argument("--max-steps", type=int, default=800)
    p_uni.add_argument("--spawn-mode", type=str, default="global")
    p_uni.add_argument("--case", type=str, choices=["min", "max", "both", "custom"], default="both")
    p_uni.add_argument("--dir", type=int, default=DIR_MIN)
    p_uni.add_argument("--range", type=int, default=RANGE_MIN)
    p_uni.add_argument("--game-mode", type=str, default="normal")
    p_uni.add_argument("--start-mode", type=str, choices=["rotate", "random"], default="rotate")
    p_uni.add_argument("--obstacle-mode", type=str, choices=["none", "random", "zones"], default="none")
    p_uni.add_argument("--n-obstacles", type=int, default=0)

    args = parser.parse_args()
    cmd = args.cmd

    if cmd is None or cmd in ("balance", "b"):
        run_balance(
            n_factions=int(getattr(args, "n_factions", 8)),
            counts=str(getattr(args, "counts", "25,23,20,18,16,14,12,10")),
            n_games=int(getattr(args, "n_games", 200)),
            board_size=int(getattr(args, "board_size", 12)),
            max_steps=int(getattr(args, "max_steps", 800)),
            spawn_mode=str(getattr(args, "spawn_mode", "zones")),
            n_rounds=int(getattr(args, "n_rounds", 10)),
            lr=float(getattr(args, "lr", 0.3)),
            game_mode=str(getattr(args, "game_mode", "normal")),
            start_mode=str(getattr(args, "start_mode", "rotate")),
            obstacle_mode=str(getattr(args, "obstacle_mode", "none")),
            n_obstacles=int(getattr(args, "n_obstacles", 0)),
        )
        return 0

    if cmd in ("uniform", "u"):
        run_uniform(
            n_factions=int(getattr(args, "n_factions", 8)),
            counts=str(getattr(args, "counts", "25,23,20,18,16,14,12,10")),
            n_games=int(getattr(args, "n_games", 800)),
            board_size=int(getattr(args, "board_size", 12)),
            max_steps=int(getattr(args, "max_steps", 800)),
            spawn_mode=str(getattr(args, "spawn_mode", "global")),
            case=str(getattr(args, "case", "both")),
            dir_count=int(getattr(args, "dir", DIR_MIN)),
            max_range=int(getattr(args, "range", RANGE_MIN)),
            game_mode=str(getattr(args, "game_mode", "normal")),
            start_mode=str(getattr(args, "start_mode", "rotate")),
            obstacle_mode=str(getattr(args, "obstacle_mode", "none")),
            n_obstacles=int(getattr(args, "n_obstacles", 0)),
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
