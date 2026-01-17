from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from experiments.lbigd.core.lbo import get_laplacian, smooth_winrate


ORTH_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
DIAG_DIRS = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
ALL_DIRS = ORTH_DIRS + DIAG_DIRS

DIR_MIN = 2
DIR_MAX = len(ALL_DIRS)
RANGE_MIN = 1
DEFAULT_UNIT_SPREAD_FRAC = 0.35
KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]


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
        
        unit_dir_spread = float(config.get("unit_dir_spread", DEFAULT_UNIT_SPREAD_FRAC * float(DIR_MAX - DIR_MIN)))
        unit_range_spread = float(config.get("unit_range_spread", DEFAULT_UNIT_SPREAD_FRAC * float(self.board_size - RANGE_MIN)))

        self.cannon_ratio: List[float] = []
        self.knight_ratio: List[float] = []
        self.unit_key: List[List[float]] = []

        self.dir_count: List[List[float]] = []
        self.max_range: List[List[float]] = []
        self.sliding: List[List[bool]] = []
        self.cannon: List[List[bool]] = []
        self.knight: List[List[bool]] = []
        for f in range(self.n_factions):
            u = self.n_units[f]
            ratio = (max_u - u) / spread
            dc_mean = float(DIR_MIN) + ratio * float(DIR_MAX - DIR_MIN)
            mr_mean = float(RANGE_MIN) + ratio * float(self.board_size - RANGE_MIN)
            cannon_ratio = float(config.get("cannon_ratio", 1.0 - ratio))
            knight_ratio = float(config.get("knight_ratio", 0.0))
            if cannon_ratio < 0.0:
                cannon_ratio = 0.0
            if knight_ratio < 0.0:
                knight_ratio = 0.0
            if cannon_ratio + knight_ratio > 1.0:
                s = float(cannon_ratio + knight_ratio)
                if s > 0.0:
                    cannon_ratio = cannon_ratio / s
                    knight_ratio = knight_ratio / s
                else:
                    cannon_ratio = 0.0
                    knight_ratio = 0.0

            self.cannon_ratio.append(float(cannon_ratio))
            self.knight_ratio.append(float(knight_ratio))

            if u <= 0:
                self.unit_key.append([])
                self.dir_count.append([])
                self.max_range.append([])
                self.sliding.append([])
                self.cannon.append([])
                self.knight.append([])
                continue

            self.unit_key.append([float(x) for x in self.rng.random(u)])

            if u == 1:
                self.dir_count.append([dc_mean])
                self.max_range.append([mr_mean])
                self.sliding.append([True])
                self.cannon.append([False])
                self.knight.append([False])
                continue

            base = np.linspace(-1.0, 1.0, u, dtype=np.float32)
            perm = np.arange(u, dtype=np.int32)
            self.rng.shuffle(perm)
            dirs = dc_mean + base * unit_dir_spread
            rngs = mr_mean + base[perm] * unit_range_spread
            self.dir_count.append([float(x) for x in dirs])
            self.max_range.append([float(x) for x in rngs])
            self.sliding.append([True] * u)
            self.cannon.append([False] * u)
            self.knight.append([False] * u)
        
        self._assign_patterns()

        self.positions: List[List[Tuple[int, int]]] = []
        self.alive: List[List[bool]] = []
        self.king_idx: List[int] = []
        self.reset()

    def _assign_patterns(self) -> None:
        for f in range(self.n_factions):
            u = len(self.unit_key[f])
            if u == 0:
                continue

            kp = float(self.knight_ratio[f])
            cp = float(self.cannon_ratio[f])
            if kp < 0.0:
                kp = 0.0
            if cp < 0.0:
                cp = 0.0
            if kp + cp > 1.0:
                s = float(kp + cp)
                if s > 0.0:
                    kp = kp / s
                    cp = cp / s
                else:
                    kp = 0.0
                    cp = 0.0
                self.knight_ratio[f] = kp
                self.cannon_ratio[f] = cp

            for i in range(u):
                r = float(self.unit_key[f][i])
                self.knight[f][i] = bool(r < kp)
                self.cannon[f][i] = bool((not self.knight[f][i]) and (r < kp + cp))

    def _dir_i(self, f: int, i: int) -> int:
        v = int(round(float(self.dir_count[f][i])))
        if v < DIR_MIN:
            return DIR_MIN
        vmax = len(ORTH_DIRS) if bool(self.cannon[f][i]) else DIR_MAX
        if v > vmax:
            return vmax
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

    def _moves_unit(
        self,
        f: int,
        i: int,
        occupied: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        if not self.alive[f][i]:
            return [], []
        x, y = self.positions[f][i]
        if bool(self.knight[f][i]):
            cap: List[Tuple[int, int, int, int]] = []
            mv: List[Tuple[int, int]] = []
            for dx, dy in KNIGHT_OFFSETS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    continue
                hit = occupied.get((nx, ny), None)
                if hit is None:
                    mv.append((nx, ny))
                    continue
                of, oi = hit
                if of != f:
                    cap.append((nx, ny, of, oi))
            return cap, mv
        dc = self._dir_i(f, i)
        mr = self._range_i(f, i)
        sl = self.sliding[f][i]
        dirs = ALL_DIRS[:dc]
        cap: List[Tuple[int, int, int, int]] = []
        mv: List[Tuple[int, int]] = []
        is_cannon = bool(self.cannon[f][i])

        for dx, dy in dirs:
            rng = range(1, mr + 1) if sl else [1]
            if not is_cannon:
                for d in rng:
                    nx, ny = x + dx * d, y + dy * d
                    if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                        break
                    hit = occupied.get((nx, ny), None)
                    if hit is None:
                        mv.append((nx, ny))
                        continue
                    of, oi = hit
                    if of != f:
                        cap.append((nx, ny, of, oi))
                    break
                continue

            screen = False
            for d in rng:
                nx, ny = x + dx * d, y + dy * d
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    break
                hit = occupied.get((nx, ny), None)
                if not screen:
                    if hit is None:
                        mv.append((nx, ny))
                        continue
                    screen = True
                    continue
                if hit is None:
                    continue
                of, oi = hit
                if of != f:
                    cap.append((nx, ny, of, oi))
                break

        return cap, mv

    def step(self, action: Tuple[int, int, int, int], occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> Tuple[Any, List[float], bool, Dict]:
        f, i, nx, ny = action
        rewards = [0.0] * self.n_factions
        if f >= self.n_factions or i >= len(self.alive[f]) or not self.alive[f][i]:
            self.step_idx += 1
            return {}, rewards, False, {}
        occ = occupied
        eliminated = None
        src = self.positions[f][i]
        if src in occ:
            del occ[src]

        hit = occ.get((nx, ny), None)
        if hit is not None:
            tf, ti = hit
            if tf == f:
                self.step_idx += 1
                occ[src] = (f, i)
                return {}, rewards, False, {}
            self.alive[tf][ti] = False
            del occ[(nx, ny)]
            if self.game_mode == "reverse":
                rewards[f] -= 1.0
                rewards[tf] += 1.0
                if sum(self.alive[tf]) == 0:
                    eliminated = tf
            else:
                rewards[f] += 1.0
                rewards[tf] -= 1.0

        self.positions[f][i] = (int(nx), int(ny))
        occ[(int(nx), int(ny))] = (f, i)
        self.step_idx += 1
        done = False
        winner = None
        if self.game_mode == "reverse":
            if eliminated is not None:
                done = True
                winner = eliminated
        else:
            if not done:
                active = [ff for ff in range(self.n_factions) if self.king_idx[ff] >= 0 and self.alive[ff][self.king_idx[ff]]]
                if len(active) == 1:
                    done = True
                    winner = active[0]
        if self.step_idx >= self.max_steps:
            done = True
            alive_counts = [sum(self.alive[ff]) for ff in range(self.n_factions)]
            if self.game_mode == "reverse":
                alive_ratio = [
                    (float(alive_counts[ff]) / float(max(1, self.n_units[ff])), ff) for ff in range(self.n_factions)
                ]
                best = min(alive_ratio)[0] if alive_ratio else 1.0
                cand = [ff for r, ff in alive_ratio if float(r) == float(best)]
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
            cap_moves: List[Tuple[int, int, int, int]] = []
            mv_moves: List[Tuple[int, int, int, int]] = []
            for i in idx:
                cap, mv = self._moves_unit(f, int(i), occ)
                for nx, ny, tf, ti in cap:
                    cap_moves.append((f, int(i), int(nx), int(ny)))
                for nx, ny in mv:
                    mv_moves.append((f, int(i), int(nx), int(ny)))

            action = None
            if cap_moves:
                action = cap_moves[int(self.rng.integers(0, len(cap_moves)))]
            elif mv_moves:
                action = mv_moves[int(self.rng.integers(0, len(mv_moves)))]

            if action is None:
                self.step_idx += 1
                continue

            _, _, done, info = self.step(action, occ)
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

                ratio_delta = float((-sign) * lr * float(err[f]))
                self.cannon_ratio[f] = float(np.clip(self.cannon_ratio[f] + ratio_delta, 0.0, 1.0))
                self.knight_ratio[f] = float(np.clip(self.knight_ratio[f] + ratio_delta, 0.0, 1.0))
                if self.cannon_ratio[f] + self.knight_ratio[f] > 1.0:
                    s = float(self.cannon_ratio[f] + self.knight_ratio[f])
                    if s > 0.0:
                        self.cannon_ratio[f] = self.cannon_ratio[f] / s
                        self.knight_ratio[f] = self.knight_ratio[f] / s
                    else:
                        self.cannon_ratio[f] = 0.0
                        self.knight_ratio[f] = 0.0

            self._assign_patterns()
        
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
                if bool(self.knight[f][i]):
                    unit_patterns.append(("knight", int(len(KNIGHT_OFFSETS)), 1))
                elif bool(self.cannon[f][i]):
                    unit_patterns.append(("cannon", self._dir_i(f, i), self._range_i(f, i)))
                else:
                    unit_patterns.append(("normal", self._dir_i(f, i), self._range_i(f, i)))
            patterns.append(unit_patterns)
        return {
            "wins": wins,
            "rates": {f: round(wr[f], 4) for f in range(self.n_factions)},
            "smoothed": [round(x, 4) for x in sm],
            "patterns": patterns,
        }
