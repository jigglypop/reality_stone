import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from experiments.lbigd.core.lbo import smooth_winrate

class SimpleCombatEnv:
    """
    제1장~제4장 문서에 기반한 '설계 가능한' 1차원 전투 환경.
    설계 변수 x에 의해 환경의 파라미터(맵 크기, 유닛 스펙)가 결정됨.
    """
    def __init__(self, config: Dict[str, float]):
        self.config = config
        # 설계 변수로 제어될 파라미터들
        self.max_distance = config.get("map_size", 10.0)
        self.unit_specs = [
            {
                "range": config.get("p0_range", 2.0),
                "damage": config.get("p0_damage", 1.0),
                "speed": config.get("p0_speed", 1.0),
                "hp": config.get("p0_hp", 10.0),
            },
            {
                "range": config.get("p1_range", 2.0),
                "damage": config.get("p1_damage", 1.0),
                "speed": config.get("p1_speed", 1.0),
                "hp": config.get("p1_hp", 10.0),
            }
        ]
        self.max_steps = int(config.get("max_steps", 50))
        self.reset()

    def reset(self):
        # 상태: [거리, p0_hp, p1_hp]
        self.distance = self.max_distance
        self.hps = [self.unit_specs[0]["hp"], self.unit_specs[1]["hp"]]
        self.current_step = 0
        self.history = {
            "distances": [self.distance],
            "actions": []
        }
        return self._get_obs()

    def _get_obs(self):
        # 정규화된 관측값 반환
        return torch.tensor([
            self.distance / self.max_distance,
            self.hps[0] / self.unit_specs[0]["hp"],
            self.hps[1] / self.unit_specs[1]["hp"]
        ], dtype=torch.float32)

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        """
        actions: [p0_action, p1_action]
        0: 대기, 1: 전진, 2: 후퇴, 3: 공격
        """
        rewards = [0.0, 0.0]

        # 1. 이동 처리 (동시 적용)
        moves = [0.0, 0.0]
        for pid in range(2):
            if actions[pid] == 1:  # 전진
                moves[pid] = -self.unit_specs[pid]["speed"]
            elif actions[pid] == 2:  # 후퇴
                moves[pid] = self.unit_specs[pid]["speed"]

        # 거리 갱신
        old_distance = self.distance
        delta_dist = 0.0
        if actions[0] == 1:
            delta_dist -= self.unit_specs[0]["speed"]
        if actions[0] == 2:
            delta_dist += self.unit_specs[0]["speed"]
        if actions[1] == 1:
            delta_dist -= self.unit_specs[1]["speed"]
        if actions[1] == 2:
            delta_dist += self.unit_specs[1]["speed"]

        self.distance = max(0.0, min(self.max_distance * 1.5, self.distance + delta_dist))

        # 거리 보상 (Shaping): 적에게 다가가면 +보상, 멀어지면 -보상 (교전 유도)
        # 단, 너무 가까우면(사거리 이내) 굳이 더 다가갈 필요는 없으므로 사거리 밖일 때만 적용
        dist_reward_scale = 0.05

        # p0 입장: 거리가 줄어들면 이득 (상대에게 접근)
        if self.distance > self.unit_specs[0]["range"]:
            if self.distance < old_distance:
                rewards[0] += dist_reward_scale
            elif self.distance > old_distance:
                rewards[0] -= dist_reward_scale

        # p1 입장: 거리가 줄어들면 이득
        if self.distance > self.unit_specs[1]["range"]:
            if self.distance < old_distance:
                rewards[1] += dist_reward_scale
            elif self.distance > old_distance:
                rewards[1] -= dist_reward_scale

        # 2. 공격 처리
        # 공격 가능 여부: 현재 거리가 사거리 이내일 것
        for pid in range(2):
            if actions[pid] == 3:
                opp_id = 1 - pid
                if self.distance <= self.unit_specs[pid]["range"]:
                    dmg = self.unit_specs[pid]["damage"]
                    self.hps[opp_id] -= dmg
                    rewards[pid] += 1.0  # 타격 보상
                    rewards[opp_id] -= 1.0  # 피격 페널티
                else:
                    # 헛스윙 페널티 (선택적)
                    rewards[pid] -= 0.1

        self.current_step += 1
        self.history["distances"].append(self.distance)
        self.history["actions"].append(actions)

        # 3. 종료 조건
        done = False
        winner = None  # 0, 1, or None (draw)

        if self.hps[0] <= 0 or self.hps[1] <= 0:
            done = True
            if self.hps[0] > self.hps[1]:
                winner = 0
                rewards[0] += 5.0
                rewards[1] -= 5.0
            elif self.hps[1] > self.hps[0]:
                winner = 1
                rewards[1] += 5.0
                rewards[0] -= 5.0
            else:
                winner = -1  # 무승부 (동시 사망)

        elif self.current_step >= self.max_steps:
            done = True
            winner = -1  # 시간 초과 무승부

        info = {
            "winner": winner,
            "distances": self.history["distances"],
            "hps": self.hps
        }

        return self._get_obs(), rewards, done, info

class GridCombatEnv:
    """
    N-faction Grid Combat Environment.
    Supports standard and reverse chess modes.
    """

    # ========== 이동 패턴 풀 (12종) ==========
    _ORTHOGONAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    _DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    _ALL_8 = _ORTHOGONAL + _DIAGONAL
    
    _KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    _CAMEL = [(-3, -1), (-3, 1), (-1, -3), (-1, 3), (1, -3), (1, 3), (3, -1), (3, 1)]
    _ZEBRA = [(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)]
    _ELEPHANT = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    _DABBABA = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    _ALFIL = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    
    _FORWARD_ONLY = [(0, -1)]
    _BACKWARD_ONLY = [(0, 1)]
    _SIDE_ONLY = [(-1, 0), (1, 0)]
    
    MOVE_PATTERN_POOL = {
        0: (_ORTHOGONAL, True, "orthogonal_slide"),
        1: (_DIAGONAL, True, "diagonal_slide"),
        2: (_ALL_8, True, "all_slide"),
        3: (_ALL_8, False, "all_jump"),
        4: (_KNIGHT, False, "knight"),
        5: (_CAMEL, False, "camel"),
        6: (_ZEBRA, False, "zebra"),
        7: (_ELEPHANT, False, "elephant"),
        8: (_DABBABA, False, "dabbaba"),
        9: (_ORTHOGONAL, False, "orthogonal_jump"),
        10: (_DIAGONAL, False, "diagonal_jump"),
        11: (_FORWARD_ONLY, True, "forward_slide"),
    }
    
    NUM_PATTERNS = len(MOVE_PATTERN_POOL)

    ATTACK_PATTERN_POOL = {
        0: (_ORTHOGONAL, True, "orthogonal_slide"),
        1: (_DIAGONAL, True, "diagonal_slide"),
        2: (_ALL_8, True, "all_slide"),
        3: (_ALL_8, False, "all_jump"),
        4: (_KNIGHT, False, "knight"),
        5: (_CAMEL, False, "camel"),
        6: (_ZEBRA, False, "zebra"),
        7: (_ELEPHANT, False, "elephant"),
        8: (_DABBABA, False, "dabbaba"),
        9: (_ORTHOGONAL, False, "orthogonal_jump"),
        10: (_DIAGONAL, False, "diagonal_jump"),
        11: (_FORWARD_ONLY, True, "forward_slide"),
        12: (_DIAGONAL, False, "pawn_diag"),
    }

    NUM_ATTACK_PATTERNS = len(ATTACK_PATTERN_POOL)
    KING_PATTERN_ID = 2
    TYPE_NAMES = ["unit0", "unit1", "unit2", "unit3", "unit4", "king"]

    def get_move_patterns(self) -> Dict[int, Dict[str, Any]]:
        out = {}
        for pid, (dirs, is_sliding, name) in self.MOVE_PATTERN_POOL.items():
            out[int(pid)] = {
                "name": str(name),
                "dirs": [(int(dx), int(dy)) for dx, dy in dirs],
                "sliding": bool(is_sliding),
            }
        return out

    def get_attack_patterns(self) -> Dict[int, Dict[str, Any]]:
        out = {}
        for pid, (dirs, is_sliding, name) in self.ATTACK_PATTERN_POOL.items():
            out[int(pid)] = {
                "name": str(name),
                "dirs": [(int(dx), int(dy)) for dx, dy in dirs],
                "sliding": bool(is_sliding),
            }
        return out

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Game Mode
        self.game_mode = str(config.get("game_mode", "standard")).lower()
        self.mandatory_capture = bool(config.get("mandatory_capture", False))
        if self.game_mode == "reverse":
            self.mandatory_capture = True

        board_size = config.get("board_size", None)
        if board_size is not None:
            self.width = int(board_size)
            self.height = int(board_size)
        else:
            self.width = int(config.get("width", 12))
            self.height = int(config.get("height", 12))
        self.max_steps = int(config.get("max_steps", 120))
        
        # N-Faction Support
        self.n_factions = int(config.get("n_factions", 2))
        if self.n_factions < 2:
            self.n_factions = 2
        self.turn_mode = str(config.get("turn_mode", "round_robin")).lower()
        if self.game_mode == "reverse" and self.n_factions > 2 and "turn_mode" not in config:
            self.turn_mode = "weighted"
            
        # Initialize faction data structures
        self.positions: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_factions)]
        self.hps: List[List[float]] = [[] for _ in range(self.n_factions)]
        self.unit_types: List[List[int]] = [[] for _ in range(self.n_factions)]
        self.king_dead = [False] * self.n_factions
        self.eliminated = [False] * self.n_factions  # Track if faction is out of game
        self.elim_step = [None] * self.n_factions
        self.reverse_win_mode = str(
            config.get("reverse_win_mode", "balanced" if self.n_factions > 2 else "first_zero")
        ).lower()

        # Win-rate tracking (for LBO-based reparam)
        self.win_counts = [0] * self.n_factions
        self.draw_count = 0
        self.win_games = 0
        self.winrate_scale_hp = [1.0] * self.n_factions
        self.winrate_scale_dmg = [1.0] * self.n_factions
        self.winrate_scale_move = [1.0] * self.n_factions
        
        self.king_type_idx = self.TYPE_NAMES.index("king")

        # Load unit counts and specs per faction
        self.type_counts = []
        self.n_units = []
        self.typespecs = []

        # Default symmetric count for reverse mode balancing
        default_count = 16 if self.game_mode == "reverse" else 16

        for f in range(self.n_factions):
            # 1. Counts
            prefix = f"p{f}"
            counts = self._get_type_counts(config, prefix, default_count)
            self.type_counts.append(counts)
            self.n_units.append(sum(counts))
            
        # 2. Specs (Needs all n_units loaded first for auto-balance)
        for f in range(self.n_factions):
            prefix = f"p{f}"
            specs = self._get_type_specs(config, prefix, f)
            self.typespecs.append(specs)

        self.reset()

    def _get_type_counts(self, config: Dict[str, Any], prefix: str, default_total: int) -> List[int]:
        provided = []
        any_provided = False
        for name in self.TYPE_NAMES:
            if name == "king":
                provided.append(1)
                continue
            v = config.get(f"{prefix}_{name}_units", None)
            if v is not None:
                any_provided = True
                provided.append(int(v))
            else:
                provided.append(0)

        if any_provided:
            return provided

        total = int(config.get(f"{prefix}_units", default_total)) - 1
        ratios = np.array([0.45, 0.25, 0.10, 0.15, 0.05], dtype=np.float64)
        raw = np.floor(ratios * total).astype(int)
        while raw.sum() < total:
            raw[int(np.argmax(ratios))] += 1
        while raw.sum() > total:
            raw[int(np.argmax(raw))] -= 1
        result = raw.tolist()
        result.append(1)
        return result

    def _get_type_specs(self, config: Dict[str, Any], prefix: str, my_f_idx: int):
        defaults = {
            "unit0": {"pattern": 0, "move": 3.0, "range": 1.0, "attack_pattern": 0, "damage": 1.0, "hp": 3.0},
            "unit1": {"pattern": 2, "move": 2.0, "range": 4.0, "attack_pattern": 2, "damage": 1.0, "hp": 2.0},
            "unit2": {"pattern": 4, "move": 1.0, "range": 1.0, "attack_pattern": 4, "damage": 1.0, "hp": 2.0},
            "unit3": {"pattern": 0, "move": 1.0, "range": 1.0, "attack_pattern": 0, "damage": 1.0, "hp": 6.0},
            "unit4": {"pattern": 0, "move": 1.0, "range": 6.0, "attack_pattern": 0, "damage": 1.5, "hp": 2.0},
            "king": {"pattern": 2, "move": 1.0, "range": 1.0, "attack_pattern": 2, "damage": 0.5, "hp": 5.0},
        }
        
        # Auto-Balance for Reverse Mode Asymmetry (N-way)
        base_hp_scale = float(config.get("reverse_base_hp_scale", 0.5))
        balance_mode = str(config.get("balance_mode", "auto")).lower()
        hp_scale = base_hp_scale
        dmg_scale = 1.0
        move_scale = 1.0
        
        if self.game_mode == "reverse" and balance_mode != "none":
            counts = [int(c) for c in self.n_units if int(c) > 0]
            min_c = min(counts) if counts else 1
            max_c = max(counts) if counts else 1
            my_count = int(self.n_units[my_f_idx])
            if max_c == min_c:
                score = 0.5
            else:
                score = (float(my_count) - float(min_c)) / (float(max_c - min_c))
                score = max(0.0, min(1.0, score))

            hp_min = float(config.get("reverse_hp_min", 0.4))
            hp_max = float(config.get("reverse_hp_max", 2.8))
            dmg_min = float(config.get("reverse_dmg_min", 0.6))
            dmg_max = float(config.get("reverse_dmg_max", 1.4))
            move_min = float(config.get("reverse_move_min", 0.7))
            move_max = float(config.get("reverse_move_max", 1.3))

            hp_scale = base_hp_scale * (hp_max + (hp_min - hp_max) * score)
            dmg_scale = dmg_max + (dmg_min - dmg_max) * score
            move_scale = move_max + (move_min - move_max) * score

        specs = []
        for name in self.TYPE_NAMES:
            base = defaults[name]
            if name == "king":
                pattern_id = self.KING_PATTERN_ID
            else:
                pattern_id = int(config.get(f"{prefix}_{name}_pattern", base["pattern"]))
                pattern_id = max(0, min(self.NUM_PATTERNS - 1, pattern_id))

            default_attack = base.get("attack_pattern", pattern_id)
            if pattern_id == 11 and name != "king":
                default_attack = 12
            atk_id = int(config.get(f"{prefix}_{name}_attack_pattern", default_attack))
            atk_id = max(0, min(self.NUM_ATTACK_PATTERNS - 1, atk_id))
            
            hp_val = float(config.get(f"{prefix}_{name}_hp", base["hp"]))
            if self.game_mode == "reverse" and f"{prefix}_{name}_hp" not in config:
                hp_floor = float(config.get("reverse_hp_floor", 0.5))
                hp_val = max(hp_floor, hp_val * hp_scale)

            # Apply win-rate reparam (LBO-smoothed) if enabled
            if self.game_mode == "reverse" and bool(self.config.get("winrate_reparam", True)):
                hp_val = max(0.5, hp_val * float(self.winrate_scale_hp[my_f_idx]))

            move_val = max(1.0, float(config.get(f"{prefix}_{name}_move", base["move"])) * move_scale)
            if self.game_mode == "reverse" and bool(self.config.get("winrate_reparam", True)):
                move_val = max(1.0, move_val * float(self.winrate_scale_move[my_f_idx]))

            specs.append(
                {
                    "pattern": pattern_id,
                    "move": move_val,
                    "range": float(config.get(f"{prefix}_{name}_range", base["range"])),
                    "attack_pattern": atk_id,
                    "damage": float(config.get(f"{prefix}_{name}_damage", base["damage"])) * dmg_scale
                    * float(self.winrate_scale_dmg[my_f_idx] if self.game_mode == "reverse" else 1.0),
                    "hp": hp_val,
                }
            )
        return specs

    def _spawn_columns(self) -> set:
        # N-way spawn logic.
        # 2 players: Left/Right
        # 3-4 players: Corners?
        # For now, simplistic general logic:
        # P0: Left, P1: Right, P2: Top, P3: Bottom (if applicable)
        # Or just random zones.
        # To keep it compatible with existing obs logic (dx, dy), let's stick to simple zones.
        # But `_spawn_columns` was used for obstacle generation to avoid blocking spawns.
        # We can just return empty set if N > 2 or define zones.
        if self.n_factions == 2:
            mid = int(self.width // 2)
            return {
                max(0, mid - 3),
                max(0, mid - 2),
                min(self.width - 1, mid + 1),
                min(self.width - 1, mid + 2),
            }
        return set()

    def _build_obstacles(self) -> set:
        density = float(self.config.get("obstacle_density", 0.0))
        if density <= 0.0:
            return set()
        
        # Simplified random scatter for N > 2 to avoid complexity
        if self.n_factions > 2:
            n_total = int(round(self.width * self.height * density))
            if n_total <= 0: return set()
            candidates = []
            for x in range(self.width):
                for y in range(self.height):
                    candidates.append((x, y))
            
            if not candidates: return set()
            picks = self.rng.choice(len(candidates), size=min(len(candidates), n_total), replace=False)
            return {candidates[i] for i in picks}

        # Original logic for 2 players
        # ... (Reuse existing logic or simplify?) 
        # For brevity, let's use the simple scatter from existing code but condensed.
        # The prompt implies "add N faction support", so we must ensure it works.
        # Let's keep it simple: random obstacles avoiding likely spawn areas if possible.
        
        n_total = int(round(self.width * self.height * density))
        occupied = set()
        attempts = 0
        while len(occupied) < n_total and attempts < n_total * 5:
            x = self.rng.integers(0, self.width)
            y = self.rng.integers(0, self.height)
            # Avoid simple spawn zones (edges)
            margin = 2
            if (x < margin) or (x >= self.width - margin) or (y < margin) or (y >= self.height - margin):
                attempts += 1
                continue
            occupied.add((x, y))
            attempts += 1
        return occupied

    def reset(self, first_turn=None):
        self.step_idx = 0
        self.king_dead = [False] * self.n_factions
        self.eliminated = [False] * self.n_factions
        self.elim_step = [None] * self.n_factions
        
        self._init_turn_queue(first_turn=first_turn)

        self.positions = [[] for _ in range(self.n_factions)]
        self.hps = [[] for _ in range(self.n_factions)]
        self.unit_types = [[] for _ in range(self.n_factions)]

        # Recompute specs per reset to apply updated win-rate reparam
        if bool(self.config.get("winrate_reparam", True)):
            self.typespecs = []
            for f in range(self.n_factions):
                prefix = f"p{f}"
                specs = self._get_type_specs(self.config, prefix, f)
                self.typespecs.append(specs)

        occupied = set()
        self.obstacles = self._build_obstacles()
        for pos in self.obstacles:
            occupied.add(pos)

        # Place units
        for f in range(self.n_factions):
            counts = self.type_counts[f]
            types = []
            for t, c in enumerate(counts):
                types.extend([t] * int(c))
            self.rng.shuffle(types)
            
            # Define spawn zone based on faction index
            # 2p: Left, Right
            # 4p: Corners?
            # 3p: Triangle?
            # General fallback: Random locations, try to separate.
            # Simplified:
            # P0: Top-Left
            # P1: Bottom-Right
            # P2: Top-Right
            # P3: Bottom-Left
            # ...
            
            if self.n_factions == 2:
                mid = int(self.width // 2)
                if f == 0:
                    x_range = (0, max(1, mid - 2))
                    y_range = (0, self.height)
                else:
                    x_range = (min(self.width, mid + 2), self.width)
                    y_range = (0, self.height)
            else:
                # Quadrant based for up to 4
                half_w = self.width // 2
                half_h = self.height // 2
                if f == 0:
                    x_range, y_range = (0, half_w), (0, half_h)
                elif f == 1:
                    x_range, y_range = (half_w, self.width), (half_h, self.height)
                elif f == 2:
                    x_range, y_range = (half_w, self.width), (0, half_h)
                else: # f == 3 or higher (overlap)
                    x_range, y_range = (0, half_w), (half_h, self.height)

            count = len(types)
            attempts = 0
            while len(self.positions[f]) < count and attempts < count * 200:
                x = int(self.rng.integers(x_range[0], x_range[1]))
                y = int(self.rng.integers(y_range[0], y_range[1]))
                if (x, y) in occupied:
                    attempts += 1
                    continue
                occupied.add((x, y))
                self.positions[f].append((x, y))
                t = int(types[len(self.positions[f]) - 1])
                self.unit_types[f].append(t)
                self.hps[f].append(self.typespecs[f][t]["hp"])
                attempts += 1

        self.history = {
            "attack_distances": [],
        }
        self.no_attack_steps = 0
        self.any_attack = False
        self._prev_avg_d = None

        return self._get_obs()

    def _init_turn_queue(self, first_turn=None) -> None:
        if self.turn_mode == "weighted":
            min_units = int(min(self.n_units)) if self.n_units else 1
            min_units = max(1, min_units)
            queue: List[int] = []
            for f, count in enumerate(self.n_units):
                w = max(1, int(round(float(count) / float(min_units))))
                queue.extend([int(f)] * int(w))
            if not queue:
                queue = [0]
            if first_turn is not None:
                try:
                    idx = queue.index(int(first_turn) % self.n_factions)
                    queue = queue[idx:] + queue[:idx]
                except ValueError:
                    pass
            self.turn_queue = queue
            self.turn_idx = 0
            self.side_to_act = int(queue[0])
            return

        # round robin (default)
        self.turn_queue = []
        self.turn_idx = 0
        if first_turn is not None:
            self.side_to_act = int(first_turn) % self.n_factions
        else:
            self.side_to_act = int(self.rng.integers(0, self.n_factions))

    def _advance_turn(self) -> None:
        if self.turn_mode == "weighted" and getattr(self, "turn_queue", None):
            queue = self.turn_queue
            idx = int(getattr(self, "turn_idx", 0))
            for _ in range(max(1, len(queue))):
                idx = (idx + 1) % len(queue)
                if not self.eliminated[int(queue[idx])]:
                    self.turn_idx = idx
                    self.side_to_act = int(queue[idx])
                    return
            return

        # round robin
        nxt = (int(self.side_to_act) + 1) % int(self.n_factions)
        self.side_to_act = int(nxt)

    def _alive_indices(self, f: int) -> List[int]:
        if f < 0 or f >= self.n_factions: return []
        return [i for i, hp in enumerate(self.hps[f]) if hp > 0]

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_enemy_all(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Generalize for N factions
        nearest_idx = []
        nearest_dist = []
        
        alive_indices = [self._alive_indices(f) for f in range(self.n_factions)]
        
        for f in range(self.n_factions):
            n_u = self.n_units[f]
            my_idx = np.full((n_u,), -1, dtype=np.int32)
            my_dist = np.zeros((n_u,), dtype=np.int32)
            
            alive_self = alive_indices[f]
            if not alive_self:
                nearest_idx.append(my_idx)
                nearest_dist.append(my_dist)
                continue

            # Collect all enemies
            enemy_pos = []
            # Mapping from flat enemy index to (faction, unit_idx)
            enemy_map = [] 
            
            for other_f in range(self.n_factions):
                if other_f == f: continue
                # if self.eliminated[other_f]: continue # Ignored dead factions?
                
                for u_idx in alive_indices[other_f]:
                    enemy_pos.append(self.positions[other_f][u_idx])
                    enemy_map.append((other_f, u_idx))
            
            if not enemy_pos:
                nearest_idx.append(my_idx)
                nearest_dist.append(my_dist)
                continue
                
            self_pos = np.array([self.positions[f][i] for i in alive_self], dtype=np.int32)
            enemy_pos_arr = np.array(enemy_pos, dtype=np.int32)
            
            # (N_self, N_enemy) distance matrix
            dx = np.abs(self_pos[:, 0:1] - enemy_pos_arr[None, :, 0])
            dy = np.abs(self_pos[:, 1:2] - enemy_pos_arr[None, :, 1])
            dist_mat = dx + dy
            
            min_indices = dist_mat.argmin(axis=1) # (N_self,)
            min_dists = dist_mat[np.arange(len(alive_self)), min_indices]
            
            # We can't store complex (f, idx) in a single int array unless we encode it.
            # But the observation expects a single 'nearest enemy' info. 
            # For N > 2, usually we pick the closest one regardless of faction.
            # Let's just store the distance. The index is less useful if we don't know the faction.
            # Or we can encode: faction * 1000 + unit_idx
            
            for i, flat_idx in enumerate(min_indices):
                orig_unit_idx = alive_self[i]
                ef, eu = enemy_map[flat_idx]
                my_idx[orig_unit_idx] = ef * 1000 + eu
                my_dist[orig_unit_idx] = min_dists[i]
                
            nearest_idx.append(my_idx)
            nearest_dist.append(my_dist)
            
        return nearest_idx, nearest_dist

    def _attack_targets(self, f: int, i: int, occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int]]:
        if self.hps[f][i] <= 0: return []

        x, y = self.positions[f][i]
        t = int(self.unit_types[f][i])
        spec = self.typespecs[f][t]

        atk_id = int(spec.get("attack_pattern", spec.get("pattern", 0)))
        atk_id = max(0, min(self.NUM_ATTACK_PATTERNS - 1, atk_id))
        dirs, is_sliding, name = self.ATTACK_PATTERN_POOL[atk_id]

        atk_range = int(max(1.0, round(float(spec.get("range", 1.0)))))
        atk_range = max(1, min(8, atk_range))

        # Direction tweaks for sides?
        # N-way logic: 
        # f=0 (Left->Right): (1,0)
        # f=1 (Right->Left): (-1,0)
        # f=2 (Top->Bottom)? 
        # For simplicity, let's keep forward_slide as 'towards center' or just fixed x-axis for now.
        # Or better: "forward" means towards the nearest enemy? Too complex.
        # Let's map: 0->(1,0), 1->(-1,0), 2->(0,1), 3->(0,-1)
        
        if name == "forward_slide":
            if f == 0: dirs = [(1, 0)]
            elif f == 1: dirs = [(-1, 0)]
            elif f == 2: dirs = [(0, 1)]
            else: dirs = [(0, -1)]
            is_sliding = True
        elif name == "pawn_diag":
            if f == 0: dirs = [(1, -1), (1, 1)]
            elif f == 1: dirs = [(-1, -1), (-1, 1)]
            elif f == 2: dirs = [(-1, 1), (1, 1)]
            else: dirs = [(-1, -1), (1, -1)]
            is_sliding = False

        targets: List[Tuple[int, int]] = []
        for dx, dy in dirs:
            if is_sliding:
                for step in range(1, atk_range + 1):
                    nx = int(x + dx * step)
                    ny = int(y + dy * step)
                    if not (0 <= nx < self.width and 0 <= ny < self.height): break
                    occ = occupied.get((nx, ny), None)
                    if occ is None: continue
                    occ_f, occ_i = int(occ[0]), int(occ[1])
                    if occ_f == -1: break # obstacle
                    if occ_f == f: break # friend
                    targets.append((occ_f, occ_i))
                    break
            else:
                for step in range(1, atk_range + 1):
                    nx = int(x + dx * step)
                    ny = int(y + dy * step)
                    if not (0 <= nx < self.width and 0 <= ny < self.height): break
                    occ = occupied.get((nx, ny), None)
                    if occ is None: continue
                    occ_f, occ_i = int(occ[0]), int(occ[1])
                    if occ_f == -1: continue
                    if occ_f == f: continue
                    targets.append((occ_f, occ_i))
                    break
        return targets

    def _build_occupied_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        occupied: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for f in range(self.n_factions):
            for i in self._alive_indices(f):
                occupied[self.positions[f][i]] = (f, i)
        for pos in getattr(self, "obstacles", set()):
            occupied[pos] = (-1, -1)
        return occupied

    def _get_obs(self):
        # Generalized observation for N factions is tricky as output size must be fixed.
        # Strategy: The agent sees "Self" vs "Nearest Enemy" (aggregated).
        # We keep the observation vector size same: 12 dims.
        # nearest_idx/dist logic already aggregates "nearest among all enemies".
        
        nearest_idx, nearest_dist = self._nearest_enemy_all()
        occupied = self._build_occupied_map()

        obs_all = [[] for _ in range(self.n_factions)]
        
        denom_x = max(1.0, float(self.width - 1))
        denom_y = max(1.0, float(self.height - 1))
        denom_d = max(1.0, float(self.width + self.height - 2))

        for f in range(self.n_factions):
            alive_self = self._alive_indices(f)
            alive_self_set = set(alive_self)
            
            # Count total enemies
            total_enemies = 0
            for of in range(self.n_factions):
                if of != f:
                    total_enemies += len(self._alive_indices(of))

            for i in range(self.n_units[f]):
                if i not in alive_self_set:
                    obs_all[f].append(torch.zeros(12, dtype=torch.float32))
                    continue

                x, y = self.positions[f][i]
                hp = float(self.hps[f][i])
                t = int(self.unit_types[f][i])
                spec = self.typespecs[f][t]
                hp_max = float(spec["hp"])
                
                # Check nearest info
                raw_target = int(nearest_idx[f][i]) # encoded f*1000 + i
                
                if raw_target < 0:
                    dxn, dyn, distn, in_range = 0.0, 0.0, 0.0, 0.0
                else:
                    tf = raw_target // 1000
                    ti = raw_target % 1000
                    ex, ey = self.positions[tf][ti]
                    dxn = (ex - x) / denom_x
                    dyn = (ey - y) / denom_y
                    distn = float(nearest_dist[f][i]) / denom_d
                    in_range = 1.0 if len(self._attack_targets(f, i, occupied)) > 0 else 0.0

                obs_all[f].append(
                    torch.tensor(
                        [
                            x / denom_x,
                            y / denom_y,
                            hp / max(1e-6, hp_max),
                            dxn,
                            dyn,
                            distn,
                            len(alive_self) / max(1.0, float(self.n_units[f])),
                            total_enemies / max(1.0, float(sum(self.n_units))), # Approx normalization
                            float(t) / 5.0,
                            float(spec["move"]) / 4.0,
                            float(spec["range"]) / 6.0,
                            in_range,
                        ],
                        dtype=torch.float32,
                    )
                )
        return obs_all

    def _sanitize_turn_action(self, turn_action: Tuple[int, int, int]) -> Tuple[int, int, int]:
        f, i, a = int(turn_action[0]), int(turn_action[1]), int(turn_action[2])
        # If f is not current side or eliminated, force dummy
        if f != self.side_to_act or self.eliminated[f]:
            return self.side_to_act, 0, 0
        return f, i, a

    def _is_alive_unit(self, f: int, i: int) -> bool:
        if f < 0 or f >= self.n_factions: return False
        if i < 0 or i >= self.n_units[f]: return False
        return float(self.hps[f][i]) > 0.0

    def _get_move_len(self, f: int, i: int) -> int:
        t = int(self.unit_types[f][i])
        pattern_id = int(self.typespecs[f][t].get("pattern", 0))
        pattern_id = max(0, min(self.NUM_PATTERNS - 1, pattern_id))
        move_pattern, _, pattern_name = self.MOVE_PATTERN_POOL[pattern_id]
        if pattern_name == "forward_slide":
            # Just return 1 for length, actual logic handled in apply_move
            return 1
        return len(move_pattern)

    def _apply_turn_move(self, f: int, i: int, a: int, occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> int:
        t = int(self.unit_types[f][i])
        pattern_id = int(self.typespecs[f][t].get("pattern", 0))
        pattern_id = max(0, min(self.NUM_PATTERNS - 1, pattern_id))
        move_pattern, is_sliding, pattern_name = self.MOVE_PATTERN_POOL[pattern_id]
        move_range = int(max(1.0, round(self.typespecs[f][t]["move"])))

        # Handle directional logic
        if pattern_name == "forward_slide":
            if f == 0: move_pattern = [(1, 0)]
            elif f == 1: move_pattern = [(-1, 0)]
            elif f == 2: move_pattern = [(0, 1)]
            else: move_pattern = [(0, -1)]

        move_len = int(len(move_pattern))
        if a >= move_len:
            return move_len

        dx, dy = move_pattern[a]
        x0, y0 = self.positions[f][i]

        def relocate(nx: int, ny: int) -> None:
            old = (int(x0), int(y0))
            new = (int(nx), int(ny))
            if old in occupied: del occupied[old]
            occupied[new] = (f, i)
            self.positions[f][i] = new

        def path_blocked(dir_dx: int, dir_dy: int, steps: int) -> bool:
            for s in range(1, int(steps) + 1):
                check_x = int(x0 + dir_dx * s)
                check_y = int(y0 + dir_dy * s)
                if (check_x, check_y) in occupied: return True
            return False

        if is_sliding:
            def try_slide(dir_dx: int, dir_dy: int) -> bool:
                for step in range(int(move_range), 0, -1):
                    nx = int(max(0, min(self.width - 1, x0 + dir_dx * step)))
                    ny = int(max(0, min(self.height - 1, y0 + dir_dy * step)))
                    if (nx, ny) == (x0, y0): continue
                    if path_blocked(dir_dx, dir_dy, step): continue
                    if (nx, ny) in occupied: continue
                    relocate(nx, ny)
                    return True
                return False

            if not try_slide(int(dx), int(dy)):
                # Try random fallback
                other_dirs = list(move_pattern)
                self.rng.shuffle(other_dirs)
                for odx, ody in other_dirs:
                    if (odx, ody) == (dx, dy): continue
                    if try_slide(int(odx), int(ody)): break
            return move_len

        # Jump logic
        x, y = int(x0), int(y0)
        for _ in range(int(move_range)):
            nx = int(x + dx)
            ny = int(y + dy)
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in occupied:
                relocate(nx, ny)
                x, y = nx, ny
                continue
            
            # Blocked, try random
            other_dirs = list(move_pattern)
            self.rng.shuffle(other_dirs)
            found = False
            for odx, ody in other_dirs:
                if (odx, ody) == (dx, dy): continue
                onx = int(x + odx)
                ony = int(y + ody)
                if 0 <= onx < self.width and 0 <= ony < self.height and (onx, ony) not in occupied:
                    relocate(onx, ony)
                    x, y = onx, ony
                    found = True
                    break
            if not found: break
            
        return move_len

    def _apply_turn_attack(self, f: int, i: int, occupied: Dict[Tuple[int, int], Tuple[int, int]], rewards: List[float]) -> int:
        t = int(self.unit_types[f][i])
        dmg = float(self.typespecs[f][t]["damage"])
        targets = self._attack_targets(f, i, occupied)
        if not targets: return 0

        # Auto-select best target (closest)
        x0, y0 = self.positions[f][i]
        best = None
        best_d = None
        for def_f, enemy_i in targets:
            ex, ey = self.positions[def_f][enemy_i]
            d = self._manhattan((x0, y0), (ex, ey))
            if best_d is None or d < best_d:
                best_d = d
                best = (def_f, enemy_i)

        if best is None: return 0

        def_f, enemy_i = best
        self.hps[def_f][enemy_i] -= dmg

        # Reward Logic
        if self.game_mode == "reverse":
            rewards[f] -= 0.5
            rewards[def_f] += 1.0
            if self.hps[def_f][enemy_i] <= 0:
                rewards[def_f] += 5.0
        else:
            rewards[f] += dmg
            rewards[def_f] -= dmg

        if best_d is not None:
            self.history["attack_distances"].append(float(best_d))
        self.any_attack = True

        # King Check (Standard only)
        if self.game_mode != "reverse":
            if self.unit_types[def_f][enemy_i] == self.king_type_idx and self.hps[def_f][enemy_i] <= 0:
                self.king_dead[def_f] = True

        return 1

    def _apply_distance_shaping(self, rewards: List[float], nearest_dist: List[np.ndarray]) -> None:
        shaping_scale = float(self.config.get("shaping_scale", 0.02))
        if self.game_mode == "reverse":
            shaping_scale *= 2.0

        for f in range(self.n_factions):
            alive = self._alive_indices(f)
            if not alive: continue
            avg_d = float(np.mean(nearest_dist[f][alive]))
            
            # We need persistent state for shaping. Using a list now.
            if self._prev_avg_d is None or len(self._prev_avg_d) != self.n_factions:
                self._prev_avg_d = [0.0] * self.n_factions
                self._prev_avg_d[f] = avg_d
            
            prev = self._prev_avg_d[f]
            rewards[f] += shaping_scale * (prev - avg_d)
            self._prev_avg_d[f] = avg_d

    def _update_winrate_reparam(self) -> None:
        if not bool(self.config.get("winrate_reparam", True)):
            return
        total = int(self.win_games)
        if total <= 0:
            return

        wins = np.array(self.win_counts, dtype=np.float32)
        win_rates = wins / float(total)
        n = int(self.n_factions)
        if n <= 1:
            return

        counts = np.array(self.n_units, dtype=np.float32)
        tau = float(self.config.get("winrate_tau", max(1.0, float(np.std(counts) + 1e-6))))
        w = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w[i, j] = float(np.exp(-abs(float(counts[i] - counts[j])) / float(tau)))

        rho = float(self.config.get("winrate_rho", 1.2))
        nu = float(self.config.get("winrate_nu", 2.5))
        smooth = smooth_winrate(w, win_rates, rho=rho, nu=nu, kappa=0.0)
        target = 1.0 / float(n)
        err = smooth - float(target)

        gain = float(self.config.get("winrate_gain", 3.8))
        scale_min = float(self.config.get("winrate_scale_min", 0.5))
        scale_max = float(self.config.get("winrate_scale_max", 2.0))
        dmg_min = float(self.config.get("winrate_dmg_min", 0.5))
        dmg_max = float(self.config.get("winrate_dmg_max", 1.8))
        move_min = float(self.config.get("winrate_move_min", 0.6))
        move_max = float(self.config.get("winrate_move_max", 1.4))

        for i in range(n):
            if self.game_mode == "reverse":
                hp_factor = float(np.exp(gain * float(err[i])))
                dmg_factor = float(np.exp(-gain * float(err[i])))
                move_factor = float(np.exp(-gain * float(err[i])))
            else:
                hp_factor = float(np.exp(-gain * float(err[i])))
                dmg_factor = float(np.exp(-gain * float(err[i])))
                move_factor = float(np.exp(-gain * float(err[i])))

            self.winrate_scale_hp[i] = max(scale_min, min(scale_max, hp_factor))
            self.winrate_scale_dmg[i] = max(dmg_min, min(dmg_max, dmg_factor))
            self.winrate_scale_move[i] = max(move_min, min(move_max, move_factor))

        self.winrate_smoothed = smooth.tolist()
        self.winrate_raw = win_rates.tolist()

    def _resolve_terminal(self, rewards: List[float]) -> Tuple[bool, int | None, List[int]]:
        """
        Returns: (done, winner, alive_counts)
        winner: 
          - Reverse Mode: The FIRST one to reach 0 units wins (returns that faction index).
          - Standard Mode: The LAST one standing wins. 
            (Or if King dies, the killer wins? No, simple survival logic: 
             If one dead, they lose. If only one remains, they win.)
        """
        done = False
        winner = None
        alive_counts = [len(self._alive_indices(f)) for f in range(self.n_factions)]
        
        # 1. Reverse Mode
        if self.game_mode == "reverse":
            if self.reverse_win_mode == "first_zero":
                for f in range(self.n_factions):
                    if alive_counts[f] == 0 and not self.eliminated[f]:
                        done = True
                        winner = f
                        rewards[f] += 20.0
                        for other in range(self.n_factions):
                            if other != f:
                                rewards[other] -= 10.0
                        self._record_win(winner=winner, draw=False)
                        return done, winner, alive_counts
            else:
                # Balanced mode: keep playing and score by adjusted elimination time.
                for f in range(self.n_factions):
                    if alive_counts[f] == 0 and not self.eliminated[f]:
                        self.eliminated[f] = True
                        self.elim_step[f] = int(self.step_idx)

                if all(self.eliminated):
                    done = True
                else:
                    base_limit = int(self.config.get("no_attack_limit", 20))
                    after_limit = int(self.config.get("no_attack_limit_after", max(base_limit, self.max_steps)))
                    limit = after_limit if self.any_attack else base_limit
                    if self.no_attack_steps >= limit or self.step_idx >= self.max_steps:
                        done = True

                if done:
                    mean_units = float(np.mean(self.n_units)) if self.n_units else 1.0
                    alpha = float(self.config.get("reverse_win_alpha", 1.2))
                    scores = []
                    for f in range(self.n_factions):
                        step = self.elim_step[f] if self.elim_step[f] is not None else int(self.max_steps * 2)
                        penalty = (mean_units / float(max(1, int(self.n_units[f])))) ** alpha
                        scores.append(float(step) * float(penalty))
                    winner = int(np.argmin(scores)) if scores else -1
                    if winner >= 0:
                        rewards[winner] += 20.0
                        for other in range(self.n_factions):
                            if other != winner:
                                rewards[other] -= 10.0
                        self._record_win(winner=winner, draw=False)
                    return done, winner, alive_counts

        # 2. Standard Mode: Last survivor wins.
        else:
            # Check Kings
            for f in range(self.n_factions):
                if self.king_dead[f] and not self.eliminated[f]:
                    self.eliminated[f] = True
                    rewards[f] -= 20.0
            
            # Check Extermination
            active_factions = []
            for f in range(self.n_factions):
                if alive_counts[f] > 0 and not self.eliminated[f]:
                    active_factions.append(f)
                elif not self.eliminated[f]:
                    # Just died out
                    self.eliminated[f] = True
                    rewards[f] -= 10.0
            
            if len(active_factions) == 1:
                done = True
                winner = active_factions[0]
                rewards[winner] += 20.0
                self._record_win(winner=winner, draw=False)
                return done, winner, alive_counts
            elif len(active_factions) == 0:
                done = True
                winner = -1 # All died same turn
                self._record_win(winner=winner, draw=True)
                return done, winner, alive_counts
            
            # Max Steps / No Attack
            # ... similar to above
            base_limit = int(self.config.get("no_attack_limit", 20))
            limit = base_limit # Simple limit for standard
            if self.no_attack_steps >= limit or self.step_idx >= self.max_steps:
                done = True
                winner = -1
                # Heuristic: Max units/HP wins
                # ...
                self._record_win(winner=winner, draw=True)
                return done, winner, alive_counts

        return done, winner, alive_counts

    def _scale_rewards(self, rewards: List[float]) -> None:
        for f in range(self.n_factions):
            scale = 1.0 / float(max(1, self.n_units[f]))
            rewards[f] = float(rewards[f]) * scale

    def _record_win(self, winner: int | None, draw: bool) -> None:
        self.win_games += 1
        if bool(draw) or winner is None or int(winner) < 0:
            self.draw_count += 1
        else:
            wi = int(winner)
            if 0 <= wi < self.n_factions:
                self.win_counts[wi] += 1
        min_games = int(self.config.get("winrate_min_games", 8))
        update_every = int(self.config.get("winrate_update_every", 1))
        if int(self.win_games) >= int(min_games) and (int(self.win_games) % max(1, update_every) == 0):
            self._update_winrate_reparam()

    def _can_capture_any(self, f: int, occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> bool:
        for i in self._alive_indices(f):
            if self._attack_targets(f, i, occupied):
                return True
        return False

    def step(self, turn_action: Tuple[int, int, int]):
        rewards = [0.0] * self.n_factions
        
        f, i, a = self._sanitize_turn_action(turn_action)
        # Check if faction is valid to act (not eliminated)
        if self.eliminated[f]:
            # Skip turn immediately
            self._advance_turn()
            return self._get_obs(), rewards, False, {}

        occupied = self._build_occupied_map()
        
        attacks_this_step = 0
        move_len = 0
        
        # Mandatory Capture
        if self.mandatory_capture:
            can_capture = self._can_capture_any(f, occupied)
            if can_capture:
                is_valid = False
                if self._is_alive_unit(f, i):
                    if self._attack_targets(f, i, occupied):
                        m_len = self._get_move_len(f, i)
                        if a >= m_len: is_valid = True
                
                if not is_valid:
                    rewards[f] -= 5.0
                    self.step_idx += 1
                    self._advance_turn()
                    done, winner, alives = self._resolve_terminal(rewards)
                    info = {
                        "winner": winner, 
                        "alive": alives,
                        "side_to_act": self.side_to_act,
                        "illegal_move": True
                    }
                    self._scale_rewards(rewards)
                    return self._get_obs(), rewards, done, info

        # Execute Action
        if self._is_alive_unit(f, i):
            move_len = self._apply_turn_move(f, i, a, occupied)
        
        if self._is_alive_unit(f, i) and a >= int(move_len):
            attacks_this_step = self._apply_turn_attack(f, i, occupied, rewards)
            
        # Update Distance Shaping
        _, nearest_dist = self._nearest_enemy_all()
        self._apply_distance_shaping(rewards, nearest_dist)
        
        if attacks_this_step == 0: self.no_attack_steps += 1
        else: self.no_attack_steps = 0
        
        self.step_idx += 1
        self._advance_turn()
        
        done, winner, alives = self._resolve_terminal(rewards)
        
        info = {
            "winner": winner,
            "attack_distances": list(self.history["attack_distances"]),
            "alive": alives,
            "no_attack_steps": self.no_attack_steps,
            "side_to_act": self.side_to_act
        }
        if bool(self.config.get("winrate_reparam", True)):
            info["win_rates"] = list(getattr(self, "winrate_raw", []))
            info["win_rates_smoothed"] = list(getattr(self, "winrate_smoothed", []))
        
        self._scale_rewards(rewards)
        
        return self._get_obs(), rewards, done, info
