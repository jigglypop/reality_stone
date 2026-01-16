from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from experiments.lbigd.core.lbo import get_laplacian, smooth_winrate


ALL_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR_MIN = 2
DIR_MAX = len(ALL_DIRS)
RANGE_MIN = 1


class GridCombatEnv:
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.n_factions = int(config.get("n_factions", 2))
        self.board_size = int(config.get("board_size", 12))
        self.max_steps = int(config.get("max_steps", 800))
        self.game_mode = str(config.get("game_mode", "normal")).lower()
        
        self.n_units = []
        for f in range(self.n_factions):
            self.n_units.append(int(config.get(f"p{f}_units", 8)))
        
        max_u = max(self.n_units)
        min_u = min(self.n_units)
        spread = max(1, max_u - min_u)
        
        unit_dir_spread = float(config.get("unit_dir_spread", 0.35 * float(DIR_MAX - DIR_MIN)))
        unit_range_spread = float(config.get("unit_range_spread", 0.35 * float(self.board_size - RANGE_MIN)))

        self.dir_count: List[List[float]] = []
        self.max_range: List[List[float]] = []
        self.sliding: List[List[bool]] = []
        for f in range(self.n_factions):
            u = self.n_units[f]
            ratio = (max_u - u) / spread
            dc_mean = float(3 + int(ratio * 5))
            mr_mean = float(3 + int(ratio * 5))

            if u <= 0:
                self.dir_count.append([])
                self.max_range.append([])
                self.sliding.append([])
                continue

            if u == 1:
                self.dir_count.append([dc_mean])
                self.max_range.append([mr_mean])
                self.sliding.append([True])
                continue

            base = np.linspace(-1.0, 1.0, u, dtype=np.float32)
            perm = np.arange(u, dtype=np.int32)
            self.rng.shuffle(perm)
            dirs = dc_mean + base * unit_dir_spread
            rngs = mr_mean + base[perm] * unit_range_spread
            self.dir_count.append([float(x) for x in dirs])
            self.max_range.append([float(x) for x in rngs])
            self.sliding.append([True] * u)
        
        self.positions: List[List[Tuple[int, int]]] = []
        self.alive: List[List[bool]] = []
        self.king_idx: List[int] = []
        self.reset()

    def _dir_i(self, f: int, i: int) -> int:
        v = int(round(float(self.dir_count[f][i])))
        if v < DIR_MIN:
            return DIR_MIN
        if v > DIR_MAX:
            return DIR_MAX
        return v

    def _range_i(self, f: int, i: int) -> int:
        v = int(round(float(self.max_range[f][i])))
        if v < RANGE_MIN:
            return RANGE_MIN
        if v > self.board_size:
            return self.board_size
        return v

    def set_uniform_pattern(self, dir_count: float, max_range: float, sliding: bool = True) -> None:
        dc = float(dir_count)
        mr = float(max_range)
        if dc < float(DIR_MIN) or dc > float(DIR_MAX):
            raise ValueError(f"dir_count {dc} must be in [{DIR_MIN}, {DIR_MAX}]")
        if mr < float(RANGE_MIN) or mr > float(self.board_size):
            raise ValueError(f"max_range {mr} must be in [{RANGE_MIN}, {self.board_size}]")

        sl = bool(sliding)
        for f in range(self.n_factions):
            for i in range(len(self.dir_count[f])):
                self.dir_count[f][i] = dc
                self.max_range[f][i] = mr
                self.sliding[f][i] = sl

    def reset(self):
        self.step_idx = 0
        self.positions = []
        self.alive = []
        self.king_idx = []
        
        total_units = int(sum(self.n_units))
        capacity = int(self.board_size * self.board_size)
        if total_units > capacity:
            raise ValueError(f"total units {total_units} exceeds board capacity {capacity}")

        spawn_mode = str(self.config.get("spawn_mode", "zones")).lower()
        if spawn_mode == "zones":
            for f in range(self.n_factions):
                zone_start = f * self.board_size // self.n_factions
                zone_end = (f + 1) * self.board_size // self.n_factions
                zone_w = int(max(0, zone_end - zone_start))
                if self.n_units[f] > zone_w * self.board_size:
                    spawn_mode = "global"
                    break

        if spawn_mode == "global":
            cells = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
            self.rng.shuffle(cells)
            cur = 0
            for f in range(self.n_factions):
                n = int(self.n_units[f])
                pos_list = cells[cur : cur + n]
                cur += n
                self.positions.append([(int(x), int(y)) for (x, y) in pos_list])
                self.alive.append([True] * n)
                self.king_idx.append(0 if n > 0 else -1)
        else:
            for f in range(self.n_factions):
                n = int(self.n_units[f])
                zone_start = f * self.board_size // self.n_factions
                zone_end = (f + 1) * self.board_size // self.n_factions
                zone_cells = [(x, y) for x in range(zone_start, zone_end) for y in range(self.board_size)]
                self.rng.shuffle(zone_cells)
                pos_list = zone_cells[:n]
                self.positions.append([(int(x), int(y)) for (x, y) in pos_list])
                self.alive.append([True] * n)
                self.king_idx.append(0 if n > 0 else -1)
        return {}

    def _alive_idx(self, f: int) -> List[int]:
        return [i for i, a in enumerate(self.alive[f]) if a]

    def _occupied(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        occ: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for ff in range(self.n_factions):
            for ii, (x, y) in enumerate(self.positions[ff]):
                if self.alive[ff][ii]:
                    occ[(x, y)] = (ff, ii)
        return occ

    def _targets(self, f: int, i: int, occupied: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        if not self.alive[f][i]:
            return []
        x, y = self.positions[f][i]
        dc = self._dir_i(f, i)
        mr = self._range_i(f, i)
        sl = self.sliding[f][i]
        dirs = ALL_DIRS[:dc]
        occ = occupied if occupied is not None else self._occupied()
        out = []
        for dx, dy in dirs:
            rng = range(1, mr + 1) if sl else [1]
            for d in rng:
                nx, ny = x + dx * d, y + dy * d
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    break
                hit = occ.get((nx, ny), None)
                if hit is None:
                    continue
                of, oi = hit
                if of != f:
                    out.append((of, oi))
                break
        return out

    def step(self, action: Tuple[int, int, int], occupied: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None) -> Tuple[Any, List[float], bool, Dict]:
        f, i, a = action
        rewards = [0.0] * self.n_factions
        if f >= self.n_factions or i >= len(self.alive[f]) or not self.alive[f][i]:
            self.step_idx += 1
            return {}, rewards, False, {}
        occ = occupied if occupied is not None else self._occupied()
        targets = self._targets(f, i, occ)
        king_killed = None
        eliminated = None
        if targets and a == 1:
            tf, ti = targets[int(self.rng.integers(0, len(targets)))]
            self.alive[tf][ti] = False
            pos = self.positions[tf][ti]
            if pos in occ:
                del occ[pos]

            if self.game_mode == "reverse":
                rewards[f] -= 1.0
                rewards[tf] += 1.0
                if sum(self.alive[tf]) == 0:
                    eliminated = tf
            else:
                rewards[f] += 1.0
                rewards[tf] -= 1.0
                if ti == self.king_idx[tf]:
                    king_killed = tf
        self.step_idx += 1
        done = False
        winner = None
        if self.game_mode == "reverse":
            if eliminated is not None:
                done = True
                winner = eliminated
        else:
            if king_killed is not None:
                done = True
                winner = f
            if not done:
                active = [ff for ff in range(self.n_factions) if self.king_idx[ff] >= 0 and self.alive[ff][self.king_idx[ff]]]
                if len(active) == 1:
                    done = True
                    winner = active[0]
        if self.step_idx >= self.max_steps:
            done = True
            alive_counts = [sum(self.alive[ff]) for ff in range(self.n_factions)]
            if self.game_mode == "reverse":
                best = int(min(alive_counts))
                cand = [ff for ff, c in enumerate(alive_counts) if int(c) == best]
                winner = int(self.rng.choice(cand)) if cand else -1
            else:
                winner = int(np.argmax(alive_counts))
        return {}, rewards, done, {"winner": winner}

    def run_game(self) -> int:
        self.reset()
        occ = self._occupied()
        for s in range(self.max_steps * self.n_factions):
            f = s % self.n_factions
            idx = self._alive_idx(f)
            if not idx:
                continue
            i = int(self.rng.choice(idx))
            t = self._targets(f, i, occ)
            a = 1 if t else 0
            _, _, done, info = self.step((f, i, a), occ)
            if done:
                return info.get("winner", -1)
        return -1

    def simulate(self, n_games: int, n_rounds: int = 10, lr: float = 0.3) -> Dict[str, Any]:
        target = 1.0 / self.n_factions
        L = get_laplacian(self.n_factions, window=min(3, self.n_factions - 1))
        sign = -1.0 if self.game_mode == "reverse" else 1.0
        
        for rd in range(n_rounds):
            wins = {f: 0 for f in range(self.n_factions)}
            for g in range(n_games):
                w = self.run_game()
                if w is not None and w >= 0:
                    wins[w] = wins.get(w, 0) + 1
            
            wr = np.array([wins.get(f, 0) for f in range(self.n_factions)], dtype=np.float32) / max(1, n_games)
            err = wr - target
            
            for f in range(self.n_factions):
                delta = float(sign * lr * float(err[f]))
                for i in range(len(self.dir_count[f])):
                    self.dir_count[f][i] = float(self.dir_count[f][i]) - delta * float(DIR_MAX - DIR_MIN)
                    self.max_range[f][i] = float(self.max_range[f][i]) - delta * float(self.board_size - RANGE_MIN)
                    if float(self.dir_count[f][i]) < float(DIR_MIN):
                        self.dir_count[f][i] = float(DIR_MIN)
                    if float(self.dir_count[f][i]) > float(DIR_MAX):
                        self.dir_count[f][i] = float(DIR_MAX)
                    if float(self.max_range[f][i]) < float(RANGE_MIN):
                        self.max_range[f][i] = float(RANGE_MIN)
                    if float(self.max_range[f][i]) > float(self.board_size):
                        self.max_range[f][i] = float(self.board_size)
        
        wins = {f: 0 for f in range(self.n_factions)}
        for g in range(n_games):
            w = self.run_game()
            if w is not None and w >= 0:
                wins[w] = wins.get(w, 0) + 1
        
        wr = np.array([wins.get(f, 0) for f in range(self.n_factions)], dtype=np.float32) / max(1, n_games)
        sm = smooth_winrate(L, wr, rho=0.5, nu=1.0)
        
        patterns = []
        for f in range(self.n_factions):
            unit_patterns = []
            for i in range(len(self.dir_count[f])):
                unit_patterns.append((self._dir_i(f, i), self._range_i(f, i)))
            patterns.append(unit_patterns)
        return {
            "wins": wins,
            "rates": {f: round(wr[f], 4) for f in range(self.n_factions)},
            "smoothed": [round(x, 4) for x in sm],
            "patterns": patterns,
        }
