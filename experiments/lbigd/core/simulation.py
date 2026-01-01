from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from experiments.lbigd.envs.simple_combat import GridCombatEnv
from experiments.lbigd.core.agent import SimplePolicy, train_one_episode, get_turn_action_with_env
from itertools import combinations

POLICY_LR = 0.01


def _set_seeds(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _first_turn(ep: int, total_episodes: int) -> int:
    # 선턴 공정성: 절반은 0이 선턴, 절반은 1이 선턴
    return 0 if int(ep) < int(total_episodes) // 2 else 1


def _train_self_play(env: GridCombatEnv, policies: List[SimplePolicy], optimizers: List[torch.optim.Optimizer], train_episodes: int) -> None:
    for ep in range(int(train_episodes)):
        train_one_episode(env, policies, optimizers, first_turn=_first_turn(ep, int(train_episodes)))


def _eval_self_play(env: GridCombatEnv, policies: List[SimplePolicy], eval_episodes: int) -> Tuple[Dict[int, int], List[float]]:
    win_counts: Dict[int, int] = {0: 0, 1: 0, -1: 0}
    all_attack_distances: List[float] = []

    with torch.no_grad():
        for ep in range(int(eval_episodes)):
            obs = env.reset(first_turn=_first_turn(ep, int(eval_episodes)))
            done = False
            while not done:
                obs0_units, obs1_units = obs
                side = int(getattr(env, "side_to_act", 0))
                if side == 0:
                    ui, a, _ = get_turn_action_with_env(policies[0], env, 0, obs0_units)
                    obs, _, done, info = env.step((0, ui, a))
                else:
                    ui, a, _ = get_turn_action_with_env(policies[1], env, 1, obs1_units)
                    obs, _, done, info = env.step((1, ui, a))

            win_counts[int(info["winner"])] += 1
            all_attack_distances.extend(info.get("attack_distances", []))

    return win_counts, all_attack_distances


def run_simulation(design_params: dict, train_episodes=50, eval_episodes=20, seed=42):
    """
    2팩션 시뮬레이션 (기존 호환)
    """
    return run_simulation_pair(design_params, (0, 1), train_episodes, eval_episodes, seed)

def run_simulation_pair(design_params: dict, factions: tuple, train_episodes=50, eval_episodes=20, seed=42):
    """
    특정 팩션 쌍 시뮬레이션
    factions: (f0, f1) 팩션 ID 쌍
    """
    _set_seeds(int(seed))
    
    env = GridCombatEnv(design_params, seed=seed, factions=factions)
    
    # 두 에이전트 초기화
    policies = [SimplePolicy(), SimplePolicy()]
    optimizers = [torch.optim.Adam(p.parameters(), lr=float(POLICY_LR)) for p in policies]
    
    # 내부 루프: 학습
    _train_self_play(env, policies, optimizers, int(train_episodes))
        
    # 평가
    win_counts, all_attack_distances = _eval_self_play(env, policies, int(eval_episodes))
    
    total = int(eval_episodes)
    f0_win_rate = win_counts[0] / total
    f1_win_rate = win_counts[1] / total
    
    if len(all_attack_distances) == 0:
        avg_distance = 0.0
        distance_std = 0.0
    else:
        avg_distance = float(np.mean(all_attack_distances))
        distance_std = float(np.std(all_attack_distances))
    
    stats = {
        "factions": factions,
        "f0_win_rate": f0_win_rate,
        "f1_win_rate": f1_win_rate,
        # 2팩션 호환 키 (디자이너/로그 출력에서 사용)
        "p0_win_rate": f0_win_rate,
        "p1_win_rate": f1_win_rate,
        "draw_rate": win_counts[-1] / total,
        "avg_distance": avg_distance,
        "distance_std": distance_std,
        "distance_samples": all_attack_distances
    }
    
    return stats

def run_simulation_all_pairs(design_params: dict, train_episodes=50, eval_episodes=20, seed=42):
    """
    모든 팩션 쌍 시뮬레이션 (다팩션)
    반환: 승률 행렬 + 통합 통계
    """
    n_factions = int(design_params.get("n_factions", 2))
    
    if n_factions == 2:
        # 2팩션은 기존 방식
        stats = run_simulation_pair(design_params, (0, 1), train_episodes, eval_episodes, seed)
        return {
            "n_factions": 2,
            "pair_stats": {(0, 1): stats},
            "win_matrix": {(0, 1): stats["f0_win_rate"], (1, 0): stats["f1_win_rate"]},
            "p0_win_rate": stats["f0_win_rate"],
            "p1_win_rate": stats["f1_win_rate"],
            "draw_rate": stats["draw_rate"],
            "avg_distance": stats["avg_distance"],
            "distance_std": stats["distance_std"],
            "distance_samples": stats["distance_samples"],
        }
    
    # 다팩션: 모든 쌍 평가
    pair_stats = {}
    win_matrix = {}
    all_distances = []
    total_draws = 0
    total_games = 0
    
    for i, j in combinations(range(n_factions), 2):
        # 각 쌍에 대해 다른 시드로 평가
        pair_seed = seed + i * 1000 + j
        stats = run_simulation_pair(design_params, (i, j), train_episodes, eval_episodes, pair_seed)
        pair_stats[(i, j)] = stats
        win_matrix[(i, j)] = stats["f0_win_rate"]
        win_matrix[(j, i)] = stats["f1_win_rate"]
        all_distances.extend(stats["distance_samples"])
        total_draws += stats["draw_rate"] * eval_episodes
        total_games += eval_episodes
    
    # 통합 통계
    avg_distance = float(np.mean(all_distances)) if all_distances else 0.0
    distance_std = float(np.std(all_distances)) if all_distances else 0.0
    
    # 2팩션 호환을 위한 기본값
    p0_win_rate = win_matrix.get((0, 1), 0.5)
    p1_win_rate = win_matrix.get((1, 0), 0.5)
    
    return {
        "n_factions": n_factions,
        "pair_stats": pair_stats,
        "win_matrix": win_matrix,
        "p0_win_rate": p0_win_rate,
        "p1_win_rate": p1_win_rate,
        "draw_rate": total_draws / total_games if total_games > 0 else 0.0,
        "avg_distance": avg_distance,
        "distance_std": distance_std,
        "distance_samples": all_distances,
    }
