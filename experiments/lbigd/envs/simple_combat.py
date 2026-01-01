import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

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
            if actions[pid] == 1: # 전진
                moves[pid] = -self.unit_specs[pid]["speed"]
            elif actions[pid] == 2: # 후퇴
                moves[pid] = self.unit_specs[pid]["speed"]
        
        # 거리 갱신
        old_distance = self.distance
        delta_dist = 0.0
        if actions[0] == 1: delta_dist -= self.unit_specs[0]["speed"]
        if actions[0] == 2: delta_dist += self.unit_specs[0]["speed"]
        if actions[1] == 1: delta_dist -= self.unit_specs[1]["speed"]
        if actions[1] == 2: delta_dist += self.unit_specs[1]["speed"]
        
        self.distance = max(0.0, min(self.max_distance * 1.5, self.distance + delta_dist))
        
        # 거리 보상 (Shaping): 적에게 다가가면 +보상, 멀어지면 -보상 (교전 유도)
        # 단, 너무 가까우면(사거리 이내) 굳이 더 다가갈 필요는 없으므로 사거리 밖일 때만 적용
        dist_reward_scale = 0.05
        
        # p0 입장: 거리가 줄어들면 이득 (상대에게 접근)
        if self.distance > self.unit_specs[0]["range"]:
            if self.distance < old_distance: rewards[0] += dist_reward_scale
            elif self.distance > old_distance: rewards[0] -= dist_reward_scale
            
        # p1 입장: 거리가 줄어들면 이득
        if self.distance > self.unit_specs[1]["range"]:
            if self.distance < old_distance: rewards[1] += dist_reward_scale
            elif self.distance > old_distance: rewards[1] -= dist_reward_scale
        
        # 2. 공격 처리
        # 공격 가능 여부: 현재 거리가 사거리 이내일 것
        for pid in range(2):
            if actions[pid] == 3:
                opp_id = 1 - pid
                if self.distance <= self.unit_specs[pid]["range"]:
                    dmg = self.unit_specs[pid]["damage"]
                    self.hps[opp_id] -= dmg
                    rewards[pid] += 1.0 # 타격 보상
                    rewards[opp_id] -= 1.0 # 피격 페널티
                else:
                    # 헛스윙 페널티 (선택적)
                    rewards[pid] -= 0.1

        self.current_step += 1
        self.history["distances"].append(self.distance)
        self.history["actions"].append(actions)

        # 3. 종료 조건
        done = False
        winner = None # 0, 1, or None (draw)
        
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
                winner = -1 # 무승부 (동시 사망)
        
        elif self.current_step >= self.max_steps:
            done = True
            winner = -1 # 시간 초과 무승부

        info = {
            "winner": winner,
            "distances": self.history["distances"],
            "hps": self.hps
        }
        
        return self._get_obs(), rewards, done, info


class GridCombatEnv:
    """
    체스보다 큰 격자 맵에서 2팩션이 다수 유닛으로 교전하는 간단 환경.

    - 맵: width x height (기본 12x12 이상 권장)
    - 유닛: 팩션별 N개 (예: 한쪽은 체스(16)보다 조금 많게, 한쪽은 더 적게)
    - 행동(유닛별): 0~4 이동(정지/상/하/좌/우), 5 공격(사거리 내 가장 가까운 적 1개)
    - 이동거리(move_range): 1스텝에서 해당 방향으로 최대 move_range만큼 이동(빈 칸일 때만)
    - 공격거리(attack_range): 맨해튼 거리 기준
    - 킹: 각 팩션 1개씩, 잡히면 즉시 패배
    - 이동 패턴: 유닛 타입별로 다름 (직선/대각선/L자/전방향)
    """

    # ========== 이동 패턴 풀 (12종) ==========
    # 각 패턴은 (dx, dy) 리스트 + 슬라이딩 여부
    # 슬라이딩=True: 해당 방향으로 move_range칸까지 슬라이딩
    # 슬라이딩=False: 정확히 그 위치로 점프 (장애물 무시)
    
    # 기본 방향
    _ORTHOGONAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 상하좌우
    _DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 대각 4방향
    _ALL_8 = _ORTHOGONAL + _DIAGONAL  # 8방향
    
    # 점프 패턴 (체스 기반)
    _KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]  # L자
    _CAMEL = [(-3, -1), (-3, 1), (-1, -3), (-1, 3), (1, -3), (1, 3), (3, -1), (3, 1)]   # 3+1 점프
    _ZEBRA = [(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)]   # 3+2 점프
    _ELEPHANT = [(-2, -2), (-2, 2), (2, -2), (2, 2)]  # 대각 2칸 점프
    _DABBABA = [(-2, 0), (2, 0), (0, -2), (0, 2)]     # 직선 2칸 점프
    _ALFIL = [(-2, -2), (-2, 2), (2, -2), (2, 2)]     # 대각 2칸 점프 (=elephant)
    
    # 특수 패턴
    _FORWARD_ONLY = [(0, -1)]  # 전진만 (폰 스타일, P0 기준)
    _BACKWARD_ONLY = [(0, 1)]  # 후진만
    _SIDE_ONLY = [(-1, 0), (1, 0)]  # 좌우만
    
    # 패턴 ID -> (방향 리스트, 슬라이딩 여부, 설명)
    MOVE_PATTERN_POOL = {
        0: (_ORTHOGONAL, True, "orthogonal_slide"),   # 룩: 직선 슬라이딩
        1: (_DIAGONAL, True, "diagonal_slide"),       # 비숍: 대각 슬라이딩
        2: (_ALL_8, True, "all_slide"),               # 퀸: 전방향 슬라이딩
        3: (_ALL_8, False, "all_jump"),               # 전방향 점프 (장애물 무시)
        4: (_KNIGHT, False, "knight"),                # 나이트: L자 점프
        5: (_CAMEL, False, "camel"),                  # 카멜: 3+1 점프
        6: (_ZEBRA, False, "zebra"),                  # 제브라: 3+2 점프
        7: (_ELEPHANT, False, "elephant"),            # 엘리펀트: 대각 2칸 점프
        8: (_DABBABA, False, "dabbaba"),              # 다바바: 직선 2칸 점프
        9: (_ORTHOGONAL, False, "orthogonal_jump"),   # 직선 점프 (장애물 무시)
        10: (_DIAGONAL, False, "diagonal_jump"),      # 대각 점프 (장애물 무시)
        11: (_FORWARD_ONLY, True, "forward_slide"),   # 전진 슬라이딩만
    }
    
    NUM_PATTERNS = len(MOVE_PATTERN_POOL)  # 12종

    # ========== 공격 패턴 풀 (LBO/최적화 대상) ==========
    # 이동 패턴과 대부분 공유하되, "pawn 대각 공격"을 추가한다.
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

    NUM_ATTACK_PATTERNS = len(ATTACK_PATTERN_POOL)  # 13종 (0..12)
    
    # 킹은 패턴 고정 (전방향 1칸)
    KING_PATTERN_ID = 2  # all_slide, move=1

    TYPE_NAMES = ["unit0", "unit1", "unit2", "unit3", "unit4", "king"]  # 5종 커스텀 + 킹

    def __init__(self, config: Dict[str, float], seed: int = 42, factions: Tuple[int, int] = (0, 1)):
        """
        factions: 이번 게임에서 싸울 2팩션의 ID (예: (0, 1), (0, 2), (1, 2))
        config에는 p0, p1, p2, ... 형태로 각 팩션 설정 가능
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.factions = factions  # 이번 게임의 두 팩션 ID

        self.width = int(config.get("width", 12))
        self.height = int(config.get("height", 12))
        self.max_steps = int(config.get("max_steps", 120))
        
        # 총 팩션 수 (config에서 자동 감지)
        self.n_factions = int(config.get("n_factions", 2))

        # 유닛 타입 6종: unit0~unit4 + king
        # 킹은 항상 1개 고정, 나머지는 config에서 가져옴
        def get_type_counts(prefix: str, default_total: int) -> List[int]:
            provided = []
            any_provided = False
            for name in self.TYPE_NAMES:
                if name == "king":
                    # 킹은 항상 1개
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

            # 기본 분배 (킹 제외한 총합)
            total = int(config.get(f"{prefix}_units", default_total)) - 1  # 킹 1개 제외
            ratios = np.array([0.45, 0.25, 0.10, 0.15, 0.05], dtype=np.float64)
            raw = np.floor(ratios * total).astype(int)
            while raw.sum() < total:
                raw[int(np.argmax(ratios))] += 1
            while raw.sum() > total:
                raw[int(np.argmax(raw))] -= 1
            result = raw.tolist()
            result.append(1)  # 킹 1개 추가
            return result

        # 이번 게임에 참여하는 두 팩션만 로드
        f0, f1 = factions
        p0_counts = get_type_counts(f"p{f0}", 16)
        p1_counts = get_type_counts(f"p{f1}", 10)

        self.type_counts = [p0_counts, p1_counts]
        self.n_units = [int(sum(p0_counts)), int(sum(p1_counts))]
        
        # 킹 인덱스 저장 (마지막 유닛이 킹)
        self.king_type_idx = self.TYPE_NAMES.index("king")
        
        # 실제 팩션 ID -> 게임 내 인덱스 (0, 1) 매핑
        self.faction_ids = factions

        # 타입별 스탯 (팩션별로 독립)
        # pattern: 이동 패턴 ID (0~11)
        # move: 이동 거리/점프 횟수
        # range: 공격 사거리 (맨해튼)
        def type_specs(prefix: str):
            # 기본값(타입별) - config에 없으면 여기 기본이 적용됨
            # unit0~unit4는 커스텀, king은 고정
            defaults = {
                "unit0": {"pattern": 0, "move": 3.0, "range": 1.0, "attack_pattern": 0, "damage": 1.0, "hp": 3.0},   # 직선 슬라이딩
                "unit1": {"pattern": 2, "move": 2.0, "range": 4.0, "attack_pattern": 2, "damage": 1.0, "hp": 2.0},   # 전방향 슬라이딩
                "unit2": {"pattern": 4, "move": 1.0, "range": 1.0, "attack_pattern": 4, "damage": 1.0, "hp": 2.0},   # 나이트 점프
                "unit3": {"pattern": 0, "move": 1.0, "range": 1.0, "attack_pattern": 0, "damage": 1.0, "hp": 6.0},   # 직선 슬라이딩 (탱크)
                "unit4": {"pattern": 0, "move": 1.0, "range": 6.0, "attack_pattern": 0, "damage": 1.5, "hp": 2.0},   # 직선 슬라이딩 (시즈)
                "king": {"pattern": 2, "move": 1.0, "range": 1.0, "attack_pattern": 2, "damage": 0.5, "hp": 5.0},    # 전방향 1칸 고정
            }

            specs = []
            for name in self.TYPE_NAMES:
                base = defaults[name]
                # 킹은 패턴 고정
                if name == "king":
                    pattern_id = self.KING_PATTERN_ID
                else:
                    pattern_id = int(config.get(f"{prefix}_{name}_pattern", base["pattern"]))
                    pattern_id = max(0, min(self.NUM_PATTERNS - 1, pattern_id))

                # 공격 패턴: 기본은 이동 패턴을 따르되, pawn(=forward_slide)은 기본 공격을 pawn_diag로 둔다.
                default_attack = base.get("attack_pattern", pattern_id)
                if pattern_id == 11 and name != "king":
                    default_attack = 12
                atk_id = int(config.get(f"{prefix}_{name}_attack_pattern", default_attack))
                atk_id = max(0, min(self.NUM_ATTACK_PATTERNS - 1, atk_id))
                
                specs.append(
                    {
                        "pattern": pattern_id,
                        "move": float(config.get(f"{prefix}_{name}_move", base["move"])),
                        "range": float(config.get(f"{prefix}_{name}_range", base["range"])),
                        "attack_pattern": atk_id,
                        "damage": float(config.get(f"{prefix}_{name}_damage", base["damage"])),
                        "hp": float(config.get(f"{prefix}_{name}_hp", base["hp"])),
                    }
                )
            return specs

        # 실제 팩션 ID로 스탯 로드
        f0, f1 = factions
        self.typespecs = [
            type_specs(f"p{f0}"),
            type_specs(f"p{f1}"),
        ]

        self.reset()

    def _spawn_columns(self) -> set:
        mid = int(self.width // 2)
        # 현재 배치 규칙과 동일한 스폰 컬럼(양 진영) 집합
        return {
            max(0, mid - 3),
            max(0, mid - 2),
            min(self.width - 1, mid + 1),
            min(self.width - 1, mid + 2),
        }

    def _build_obstacles(self) -> set:
        density = float(self.config.get("obstacle_density", 0.0))
        pattern = int(self.config.get("obstacle_pattern", 0))
        if density <= 0.0 or pattern <= 0:
            return set()

        # 과도한 장애물은 배치/이동을 붕괴시키므로 상한을 둔다
        density = max(0.0, min(0.35, density))
        n_total = int(round(self.width * self.height * density))
        if n_total <= 0:
            return set()

        banned_cols = self._spawn_columns()
        max_y = int(self.height)
        max_x = int(self.width)

        # (공정성) 좌우 대칭 배치가 기본. 중앙 컬럼(홀수 폭)은 가능하면 비운다.
        def mirror_x(x: int) -> int:
            return (max_x - 1) - x

        def symmetric_scatter(candidates: list, total: int) -> set:
            if not candidates or total <= 0:
                return set()
            pairs = min(len(candidates), max(1, int(total) // 2))
            picks = self.rng.choice(len(candidates), size=pairs, replace=False)
            out = set()
            for idx in picks:
                x, y = candidates[int(idx)]
                mx = mirror_x(int(x))
                out.add((int(x), int(y)))
                out.add((int(mx), int(y)))
            return out

        def laplace_2d(z: np.ndarray) -> np.ndarray:
            """
            2D 4-neighbor discrete Laplacian (no wrap).  z shape: (w, h)
            """
            lap = (-4.0 * z).astype(np.float32, copy=False)
            lap[1:, :] += z[:-1, :]
            lap[:-1, :] += z[1:, :]
            lap[:, 1:] += z[:, :-1]
            lap[:, :-1] += z[:, 1:]
            return lap

        # pattern 1: 랜덤 대칭 산포(전역)
        def build_pattern_1() -> set:
            half = max_x // 2
            candidates = []
            for x in range(half):
                mx = mirror_x(x)
                if x in banned_cols or mx in banned_cols:
                    continue
                for y in range(max_y):
                    candidates.append((x, y))
            return symmetric_scatter(candidates, n_total)

        # pattern 2: 중앙 장벽(벽) + 랜덤 홀
        def build_pattern_2() -> set:
            mid = int(max_x // 2)
            wall_cols = [mid - 1, mid] if (max_x % 2 == 0) else [mid]
            wall_cols = [c for c in wall_cols if 0 <= c < max_x and c not in banned_cols]
            if not wall_cols:
                return build_pattern_1()

            # 벽을 전부 막으면 퇴화하기 쉬우므로, 일부 행을 비워 "통로"를 만든다.
            gap_rows_cnt = max(2, int(max_y // 4))
            gap_rows_cnt = min(gap_rows_cnt, max_y)
            gap_rows = set(self.rng.choice(max_y, size=gap_rows_cnt, replace=False).tolist())

            cells = []
            for x in wall_cols:
                for y in range(max_y):
                    if y in gap_rows:
                        continue
                    cells.append((x, y))

            if not cells:
                return build_pattern_1()

            k = min(len(cells), n_total)
            picks = self.rng.choice(len(cells), size=k, replace=False)
            out = set()
            for idx in picks:
                x, y = cells[int(idx)]
                out.add((int(x), int(y)))
            return out

        # pattern 3: 중앙 집중(센터 사각 영역) 대칭 산포
        def build_pattern_3() -> set:
            x0 = int(max(0, max_x // 4))
            x1 = int(min(max_x, max_x - max_x // 4))
            y0 = int(max(0, max_y // 4))
            y1 = int(min(max_y, max_y - max_y // 4))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                return build_pattern_1()

            half = max_x // 2
            candidates = []
            for x in range(x0, min(x1, half)):
                mx = mirror_x(x)
                if x in banned_cols or mx in banned_cols:
                    continue
                for y in range(y0, y1):
                    candidates.append((x, y))
            if not candidates:
                return build_pattern_1()
            return symmetric_scatter(candidates, n_total)

        # pattern 4: "주름/수로" 느낌의 구조를 만드는 간단한 곡률 기반 패턴
        # - 랜덤 노이즈를 확산(라플라시안)으로 저주파화
        # - |라플라시안|이 큰 곳(곡률/경계)을 장애물로 선택
        # - 좌우 대칭을 유지하고 스폰 컬럼을 피한다
        def build_pattern_4() -> set:
            half = max_x // 2
            if half <= 0:
                return build_pattern_1()

            z = self.rng.random((half, max_y), dtype=np.float32)
            diff = 0.25
            steps = 10
            for _ in range(int(steps)):
                z = (z + float(diff) * laplace_2d(z)).astype(np.float32, copy=False)

            curv = np.abs(laplace_2d(z))
            # banned cols는 점수에서 제외
            for x in range(half):
                mx = mirror_x(int(x))
                if x in banned_cols or mx in banned_cols:
                    curv[int(x), :] = -1e9

            def has_path(obstacles: set) -> bool:
                # 스폰 컬럼(좌 2열 -> 우 2열) 사이에 통로가 하나도 없으면 "퇴화"로 간주하고 완화한다.
                mid = int(max_x // 2)
                start_cols = [max(0, mid - 3), max(0, mid - 2)]
                goal_cols = [min(max_x - 1, mid + 1), min(max_x - 1, mid + 2)]
                blocked = set(obstacles)

                goals = {(int(x), int(y)) for x in goal_cols for y in range(max_y) if (int(x), int(y)) not in blocked}
                if not goals:
                    return False

                q = []
                seen = set()
                for x in start_cols:
                    for y in range(max_y):
                        p = (int(x), int(y))
                        if p in blocked:
                            continue
                        q.append(p)
                        seen.add(p)

                if not q:
                    return False

                # BFS 4-neighbor
                head = 0
                while head < len(q):
                    x, y = q[head]
                    head += 1
                    if (x, y) in goals:
                        return True
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = int(x + dx)
                        ny = int(y + dy)
                        if not (0 <= nx < max_x and 0 <= ny < max_y):
                            continue
                        np_ = (nx, ny)
                        if np_ in blocked or np_ in seen:
                            continue
                        seen.add(np_)
                        q.append(np_)
                return False

            def build_with_pairs(pairs: int) -> set:
                flat = curv.reshape(-1)
                pairs = max(1, min(int(pairs), int(flat.size)))

                # fold를 "선"처럼 만들기 위해, 상위 후보를 넉넉히 뽑고 세로 방향으로 짧은 세그먼트를 그리며 고른다.
                budget = min(int(flat.size), int(pairs) * 12)
                idx = np.argpartition(flat, -budget)[-budget:]
                idx = idx[np.argsort(flat[idx])[::-1]]

                out_half = set()
                for ii in idx.tolist():
                    if len(out_half) >= pairs:
                        break
                    x, y = np.unravel_index(int(ii), curv.shape)
                    x = int(x)
                    y = int(y)
                    if float(curv[x, y]) < -1e8:
                        continue

                    # 이미 바로 근처에 있으면 과밀해지므로 스킵(주름이 너무 점상으로 찢어지는 것 방지)
                    if (x, y) in out_half or (x, y - 1) in out_half or (x, y + 1) in out_half:
                        continue

                    for dy in (-1, 0, 1):
                        if len(out_half) >= pairs:
                            break
                        yy = int(y + dy)
                        if 0 <= yy < max_y:
                            out_half.add((x, yy))

                out = set()
                for x, y in out_half:
                    mx = mirror_x(int(x))
                    out.add((int(x), int(y)))
                    out.add((int(mx), int(y)))
                return out

            def carve_min_open(obstacles: set) -> set:
                """
                막힌 경우, "최소한의 장애물 제거"로 좌->우 연결 통로를 강제로 만든다(0-1 BFS).
                대칭을 유지하기 위해 제거는 항상 미러까지 같이 제거한다.
                """
                mid = int(max_x // 2)
                start_cols = [max(0, mid - 3), max(0, mid - 2)]
                goal_cols = [min(max_x - 1, mid + 1), min(max_x - 1, mid + 2)]
                blocked = set(obstacles)

                goals = {(int(x), int(y)) for x in goal_cols for y in range(max_y)}
                starts = [(int(x), int(y)) for x in start_cols for y in range(max_y) if (int(x), int(y)) not in blocked]
                if not starts:
                    return obstacles

                # 0-1 BFS (cost=1 if cell is blocked, else 0)
                from collections import deque

                INF = 10**9
                dist = {}
                prev = {}
                dq = deque()
                for s in starts:
                    dist[s] = 0
                    prev[s] = None
                    dq.appendleft(s)

                best_goal = None
                while dq:
                    p = dq.popleft()
                    d = dist.get(p, INF)
                    if p in goals:
                        best_goal = p
                        break
                    x, y = int(p[0]), int(p[1])
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = int(x + dx)
                        ny = int(y + dy)
                        if not (0 <= nx < max_x and 0 <= ny < max_y):
                            continue
                        np_ = (nx, ny)
                        cost = 1 if np_ in blocked else 0
                        nd = int(d + cost)
                        if nd < int(dist.get(np_, INF)):
                            dist[np_] = nd
                            prev[np_] = p
                            if cost == 0:
                                dq.appendleft(np_)
                            else:
                                dq.append(np_)

                if best_goal is None:
                    return obstacles

                # reconstruct path and remove blocked cells along it (+ mirror)
                to_remove = set()
                cur = best_goal
                while cur is not None:
                    if cur in blocked:
                        to_remove.add(cur)
                        to_remove.add((mirror_x(int(cur[0])), int(cur[1])))
                    cur = prev.get(cur, None)

                if not to_remove:
                    return obstacles
                return set(obstacles) - to_remove

            base_pairs = max(1, int(n_total) // 2)
            out = build_with_pairs(base_pairs)
            if has_path(out):
                return out
            out2 = carve_min_open(out)
            return out2

        if pattern == 2:
            return build_pattern_2()
        if pattern == 3:
            return build_pattern_3()
        if pattern == 4:
            return build_pattern_4()
        return build_pattern_1()

    def reset(self, first_turn=None):
        """
        first_turn: None이면 랜덤, 0 또는 1이면 해당 진영 선턴
        """
        self.step_idx = 0
        # first_turn이 명시되면 그대로, 아니면 랜덤
        if first_turn is not None:
            self.side_to_act = int(first_turn)
        else:
            self.side_to_act = int(self.rng.integers(0, 2))

        # positions[f] = [(x,y), ...], hps[f] = [hp, ...]
        self.positions: List[List[Tuple[int, int]]] = [[], []]
        self.hps: List[List[float]] = [[], []]
        self.unit_types: List[List[int]] = [[], []]  # 타입 인덱스
        self.king_dead = [False, False]  # 킹 사망 플래그

        occupied = set()

        # 장애물(정적 블록) 생성: 유닛 배치 전에 occupied에 넣어 충돌을 막는다
        self.obstacles = self._build_obstacles()
        for pos in self.obstacles:
            occupied.add(pos)

        # 턴제(턴당 1유닛)에서는 가장자리 배치(좌 2열 vs 우 2열)가 교전까지 너무 오래 걸려
        # 무승부로 퇴화하기 쉽습니다. 맵은 크게 유지하되, 초반 교전이 가능하도록 중앙 근처에 배치합니다.
        def place_units(f: int):
            mid = int(self.width // 2)
            if f == 0:
                cols = [max(0, mid - 3), max(0, mid - 2)]
            else:
                cols = [min(self.width - 1, mid + 1), min(self.width - 1, mid + 2)]
            counts = self.type_counts[f]
            # 타입 배열 생성 후 섞기(배치 랜덤)
            types = []
            for t, c in enumerate(counts):
                types.extend([t] * int(c))
            self.rng.shuffle(types)
            count = len(types)
            attempts = 0
            while len(self.positions[f]) < count and attempts < max(1, count) * 200:
                x = int(self.rng.choice(cols))
                y = int(self.rng.integers(0, self.height))
                if (x, y) in occupied:
                    attempts += 1
                    continue
                occupied.add((x, y))
                self.positions[f].append((x, y))
                t = int(types[len(self.positions[f]) - 1])
                self.unit_types[f].append(t)
                self.hps[f].append(self.typespecs[f][t]["hp"])
                attempts += 1

        place_units(0)
        place_units(1)

        self.history = {
            "attack_distances": [],  # 교전 거리 표본(맨해튼)
        }
        self.no_attack_steps = 0
        self.any_attack = False
        self._prev_avg_d = None

        return self._get_obs()

    def _alive_indices(self, f: int) -> List[int]:
        return [i for i, hp in enumerate(self.hps[f]) if hp > 0]

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_enemy_all(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        각 팩션 유닛 i에 대해 '가장 가까운 적'의 (enemy_index, manhattan_distance)를 벡터화로 계산합니다.
        반환:
          nearest_idx[f]: shape (n_units_f,), dead/적없음은 -1
          nearest_dist[f]: shape (n_units_f,), dead/적없음은 0
        """
        nearest_idx = [np.full((self.n_units[0],), -1, dtype=np.int32), np.full((self.n_units[1],), -1, dtype=np.int32)]
        nearest_dist = [np.zeros((self.n_units[0],), dtype=np.int32), np.zeros((self.n_units[1],), dtype=np.int32)]

        for f in (0, 1):
            opp = 1 - f
            alive_self = np.array(self._alive_indices(f), dtype=np.int32)
            alive_opp = np.array(self._alive_indices(opp), dtype=np.int32)
            if alive_self.size == 0 or alive_opp.size == 0:
                continue

            self_pos = np.array([self.positions[f][i] for i in alive_self], dtype=np.int32)  # (A,2)
            opp_pos = np.array([self.positions[opp][j] for j in alive_opp], dtype=np.int32)  # (B,2)

            # manhattan distance matrix (A,B)
            dx = np.abs(self_pos[:, 0:1] - opp_pos[None, :, 0])
            dy = np.abs(self_pos[:, 1:2] - opp_pos[None, :, 1])
            dist = dx + dy

            argmin = dist.argmin(axis=1)
            dmin = dist[np.arange(dist.shape[0]), argmin]

            nearest_idx[f][alive_self] = alive_opp[argmin]
            nearest_dist[f][alive_self] = dmin.astype(np.int32)

        return nearest_idx, nearest_dist

    def _attack_targets(self, f: int, i: int, occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        현재 유닛(f,i)이 공격 가능한 (def_f, enemy_i) 목록을 반환합니다.
        공격 패턴/사거리/장애물(시야) 규칙을 반영합니다.
        """
        if self.hps[f][i] <= 0:
            return []

        x, y = self.positions[f][i]
        t = int(self.unit_types[f][i])
        spec = self.typespecs[f][t]

        atk_id = int(spec.get("attack_pattern", spec.get("pattern", 0)))
        atk_id = max(0, min(self.NUM_ATTACK_PATTERNS - 1, atk_id))
        dirs, is_sliding, name = self.ATTACK_PATTERN_POOL[atk_id]

        atk_range = int(max(1.0, round(float(spec.get("range", 1.0)))))
        atk_range = max(1, min(8, atk_range))

        # 방향 보정
        if name == "forward_slide":
            dirs = [(1, 0)] if f == 0 else [(-1, 0)]
            is_sliding = True
        elif name == "pawn_diag":
            dirs = [(1, -1), (1, 1)] if f == 0 else [(-1, -1), (-1, 1)]
            is_sliding = False

        targets: List[Tuple[int, int]] = []
        for dx, dy in dirs:
            if is_sliding:
                for step in range(1, atk_range + 1):
                    nx = int(x + dx * step)
                    ny = int(y + dy * step)
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        break
                    occ = occupied.get((nx, ny), None)
                    if occ is None:
                        continue
                    occ_f, occ_i = int(occ[0]), int(occ[1])
                    if occ_f == -1:
                        break
                    if occ_f == f:
                        break
                    targets.append((occ_f, occ_i))
                    break
            else:
                for step in range(1, atk_range + 1):
                    nx = int(x + dx * step)
                    ny = int(y + dy * step)
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        break
                    occ = occupied.get((nx, ny), None)
                    if occ is None:
                        continue
                    occ_f, occ_i = int(occ[0]), int(occ[1])
                    if occ_f == -1:
                        continue
                    if occ_f == f:
                        continue
                    targets.append((occ_f, occ_i))
                    break

        return targets

    def _build_occupied_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        occupied: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for ff in (0, 1):
            for ii in self._alive_indices(ff):
                occupied[self.positions[ff][ii]] = (ff, ii)
        for pos in getattr(self, "obstacles", set()):
            occupied[pos] = (-1, -1)
        return occupied

    def _get_unit_obs(self, f: int, i: int) -> torch.Tensor:
        """
        단일 유닛 관측을 반환합니다.

        - 현재 정책/학습 루프는 `_get_obs()`가 만드는 12차원 유닛 관측을 사용합니다.
        - 과거 구현의 깨진 참조(self.specs, _nearest_enemy)를 제거하고, DRY하게 `_get_obs()` 결과를 재사용합니다.
        """
        try:
            ff = int(f)
            ii = int(i)
        except Exception:
            return torch.zeros(12, dtype=torch.float32)

        if ff not in (0, 1):
            return torch.zeros(12, dtype=torch.float32)

        obs_all = self._get_obs()
        if ii < 0 or ii >= len(obs_all[ff]):
            return torch.zeros(12, dtype=torch.float32)
        return obs_all[ff][ii]

    def _get_obs(self):
        # 파이썬 루프 최소화: nearest를 한 번에 계산하고 obs를 구성
        nearest_idx, nearest_dist = self._nearest_enemy_all()
        occupied = self._build_occupied_map()

        obs_all = [[], []]
        for f in (0, 1):
            opp = 1 - f
            denom_x = max(1.0, float(self.width - 1))
            denom_y = max(1.0, float(self.height - 1))
            denom_d = max(1.0, float(self.width + self.height - 2))

            alive_self = self._alive_indices(f)
            alive_self_set = set(alive_self)

            for i in range(self.n_units[f]):
                if i not in alive_self_set:
                    obs_all[f].append(torch.zeros(12, dtype=torch.float32))
                    continue

                x, y = self.positions[f][i]
                hp = float(self.hps[f][i])
                t = int(self.unit_types[f][i])
                spec = self.typespecs[f][t]
                hp_max = float(spec["hp"])
                ei = int(nearest_idx[f][i])
                if ei < 0:
                    dxn = 0.0
                    dyn = 0.0
                    distn = 0.0
                    in_range = 0.0
                else:
                    ex, ey = self.positions[opp][ei]
                    dxn = (ex - x) / denom_x
                    dyn = (ey - y) / denom_y
                    distn = float(nearest_dist[f][i]) / denom_d
                    # 공격 패턴 기준 "공격 가능 여부"를 계산한다.
                    in_range = 1.0 if len(self._attack_targets(f, i, occupied)) > 0 else 0.0

                alive_self_cnt = len(alive_self)
                alive_enemy_cnt = len(self._alive_indices(opp))

                # 타입/스탯 피처(정규화)
                # unit0~unit4 + king(=5)까지 포함하므로 0..1로 정규화
                type_norm = float(t) / 5.0
                move_norm = float(spec["move"]) / 4.0
                range_norm = float(spec["range"]) / 6.0

                obs_all[f].append(
                    torch.tensor(
                        [
                            x / denom_x,
                            y / denom_y,
                            hp / max(1e-6, hp_max),
                            dxn,
                            dyn,
                            distn,
                            alive_self_cnt / max(1.0, float(self.n_units[f])),
                            alive_enemy_cnt / max(1.0, float(self.n_units[opp])),
                            type_norm,
                            move_norm,
                            range_norm,
                            in_range,
                        ],
                        dtype=torch.float32,
                    )
                )

        return obs_all

    def _sanitize_turn_action(self, turn_action: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        규칙 위반 입력을 최소한의 방식으로 정리합니다.

        - 현재 턴의 진영이 아니면: (side_to_act, 0, 0)으로 강제
        """
        f, i, a = int(turn_action[0]), int(turn_action[1]), int(turn_action[2])
        side = int(getattr(self, "side_to_act", 0))
        if f != side:
            return side, 0, 0
        return f, i, a

    def _is_alive_unit(self, f: int, i: int) -> bool:
        if f not in (0, 1):
            return False
        if i < 0 or i >= int(self.n_units[f]):
            return False
        return float(self.hps[f][i]) > 0.0

    def _apply_turn_move(self, f: int, i: int, a: int, occupied: Dict[Tuple[int, int], Tuple[int, int]]) -> int:
        """
        현재 턴 유닛 1개의 이동(또는 정지)을 처리합니다.

        반환값은 "현재 유닛의 이동 패턴 길이"이며, step()에서 공격 판정(a >= len(move_pattern))에 사용합니다.
        """
        t = int(self.unit_types[f][i])
        pattern_id = int(self.typespecs[f][t].get("pattern", 0))
        pattern_id = max(0, min(self.NUM_PATTERNS - 1, pattern_id))
        move_pattern, is_sliding, pattern_name = self.MOVE_PATTERN_POOL[pattern_id]
        move_range = int(max(1.0, round(self.typespecs[f][t]["move"])))

        # "forward_slide"는 팩션 진행 방향(좌↔우)을 따른다.
        if pattern_name == "forward_slide":
            move_pattern = [(1, 0)] if f == 0 else [(-1, 0)]

        move_len = int(len(move_pattern))
        if a >= move_len:
            return move_len

        dx, dy = move_pattern[a]
        x0, y0 = self.positions[f][i]

        def relocate(nx: int, ny: int) -> None:
            old = (int(x0), int(y0))
            new = (int(nx), int(ny))
            # step()의 호출자는 occupied 충돌을 미리 체크한다.
            if old in occupied:
                del occupied[old]
            occupied[new] = (f, i)
            self.positions[f][i] = new

        def path_blocked(dir_dx: int, dir_dy: int, steps: int) -> bool:
            # 기존 구현과 동일: 중간 경로에 occupied(유닛/장애물)가 있으면 blocked로 본다.
            for s in range(1, int(steps) + 1):
                check_x = int(x0 + dir_dx * s)
                check_y = int(y0 + dir_dy * s)
                if (check_x, check_y) in occupied:
                    return True
            return False

        if is_sliding:
            # 슬라이딩: 해당 방향으로 move_range칸까지, 경로 상 장애물 체크
            def try_slide(dir_dx: int, dir_dy: int) -> bool:
                for step in range(int(move_range), 0, -1):
                    nx = int(max(0, min(self.width - 1, x0 + dir_dx * step)))
                    ny = int(max(0, min(self.height - 1, y0 + dir_dy * step)))
                    if (nx, ny) == (x0, y0):
                        continue
                    if path_blocked(dir_dx, dir_dy, step):
                        continue
                    if (nx, ny) in occupied:
                        continue
                    relocate(nx, ny)
                    return True
                return False

            moved = try_slide(int(dx), int(dy))
            if not moved:
                other_dirs = list(move_pattern)
                self.rng.shuffle(other_dirs)
                for odx, ody in other_dirs:
                    if (int(odx), int(ody)) == (int(dx), int(dy)):
                        continue
                    if try_slide(int(odx), int(ody)):
                        break
            return move_len

        # 점프: 정확히 (dx, dy) 위치로 이동, 장애물 무시
        x, y = int(x0), int(y0)
        for _ in range(int(move_range)):
            nx = int(x + dx)
            ny = int(y + dy)
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in occupied:
                # 기존 위치 제거 후 새 위치 등록
                if (x, y) in occupied:
                    del occupied[(x, y)]
                occupied[(nx, ny)] = (f, i)
                self.positions[f][i] = (nx, ny)
                x, y = nx, ny
                continue

            # 원래 방향 막히면 다른 방향 시도
            other_dirs = list(move_pattern)
            self.rng.shuffle(other_dirs)
            tried_other = False
            for odx, ody in other_dirs:
                if (int(odx), int(ody)) == (int(dx), int(dy)):
                    continue
                onx = int(x + odx)
                ony = int(y + ody)
                if 0 <= onx < self.width and 0 <= ony < self.height and (onx, ony) not in occupied:
                    if (x, y) in occupied:
                        del occupied[(x, y)]
                    occupied[(onx, ony)] = (f, i)
                    self.positions[f][i] = (onx, ony)
                    x, y = onx, ony
                    tried_other = True
                    break
            if not tried_other:
                break

        return move_len

    def _apply_turn_attack(self, f: int, i: int, occupied: Dict[Tuple[int, int], Tuple[int, int]], rewards: List[float]) -> int:
        """
        현재 턴 유닛 1개의 공격을 처리합니다.
        반환: 이번 step에서 공격이 실제로 발생했는지(0 또는 1)
        """
        t = int(self.unit_types[f][i])
        dmg = float(self.typespecs[f][t]["damage"])
        targets = self._attack_targets(f, i, occupied)
        if not targets:
            return 0

        # 가장 가까운(맨해튼) 적을 자동 선택
        x0, y0 = self.positions[f][i]
        best = None
        best_d = None
        for def_f, enemy_i in targets:
            ex, ey = self.positions[def_f][enemy_i]
            d = self._manhattan((x0, y0), (ex, ey))
            if best_d is None or d < best_d:
                best_d = d
                best = (def_f, enemy_i)

        if best is None or best_d is None:
            return 0

        def_f, enemy_i = int(best[0]), int(best[1])
        self.hps[def_f][enemy_i] -= dmg
        rewards[f] += dmg
        rewards[def_f] -= dmg
        self.history["attack_distances"].append(float(best_d))
        self.any_attack = True

        # 킹 사망 체크: 공격받은 유닛이 킹이고 HP <= 0이면 즉시 패배
        if self.unit_types[def_f][enemy_i] == self.king_type_idx and self.hps[def_f][enemy_i] <= 0:
            self.king_dead[def_f] = True

        return 1

    def _apply_distance_shaping(self, rewards: List[float], nearest_dist: List[np.ndarray]) -> None:
        """
        접근 유도 shaping: 살아있는 유닛들의 최근접 적 거리 평균을 줄이면 보상(+), 늘리면 패널티(-).
        """
        shaping_scale = float(self.config.get("shaping_scale", 0.02))
        for ff in (0, 1):
            alive = self._alive_indices(ff)
            if len(alive) == 0:
                continue
            avg_d = float(np.mean(nearest_dist[ff][alive])) if len(alive) else 0.0
            if self._prev_avg_d is None:
                self._prev_avg_d = [avg_d, avg_d]
            prev = self._prev_avg_d
            rewards[ff] += shaping_scale * (prev[ff] - avg_d)
            prev[ff] = avg_d
            self._prev_avg_d = prev

    def _resolve_terminal(self, rewards: List[float]) -> Tuple[bool, int | None, int, int]:
        done = False
        winner: int | None = None
        alive0 = len(self._alive_indices(0))
        alive1 = len(self._alive_indices(1))

        # 킹 사망 체크: 킹이 죽으면 즉시 패배 (체스처럼)
        if getattr(self, "king_dead", [False, False])[0]:
            done = True
            winner = 1
            rewards[1] += 20.0
            rewards[0] -= 20.0
            return done, winner, alive0, alive1
        if getattr(self, "king_dead", [False, False])[1]:
            done = True
            winner = 0
            rewards[0] += 20.0
            rewards[1] -= 20.0
            return done, winner, alive0, alive1

        if alive0 == 0 or alive1 == 0:
            done = True
            if alive0 > alive1:
                winner = 0
                rewards[0] += 10.0
                rewards[1] -= 10.0
            elif alive1 > alive0:
                winner = 1
                rewards[1] += 10.0
                rewards[0] -= 10.0
            else:
                winner = -1
            return done, winner, alive0, alive1

        # 무교전이 일정 턴 지속되면 조기 종료 + 큰 페널티(퇴화 방지)
        # - 교전이 시작된 뒤에는(=any_attack) 잠깐의 재배치/추격이 생길 수 있으므로 limit을 더 크게 둬서 "교전이 이어지게" 한다.
        base_limit = int(self.config.get("no_attack_limit", 20))
        # 기본은 "교전이 한 번이라도 발생하면 max_steps까지는 계속 진행"에 가깝게 둔다.
        # (교전 도중 잠깐의 재배치/추격 구간으로 인해 조기 종료되는 것을 방지)
        after_limit_default = int(max(base_limit, int(self.max_steps)))
        after_limit = int(self.config.get("no_attack_limit_after", after_limit_default))
        no_attack_limit = after_limit if bool(self.any_attack) else base_limit
        if self.no_attack_steps >= int(no_attack_limit):
            done = True
            if not self.any_attack:
                winner = -1
                rewards[0] -= 2.0
                rewards[1] -= 2.0
            else:
                hp0 = float(sum(max(0.0, hp) for hp in self.hps[0]))
                hp1 = float(sum(max(0.0, hp) for hp in self.hps[1]))
                if alive0 > alive1 or (alive0 == alive1 and hp0 > hp1):
                    winner = 0
                    rewards[0] += 1.0
                    rewards[1] -= 1.0
                elif alive1 > alive0 or (alive0 == alive1 and hp1 > hp0):
                    winner = 1
                    rewards[1] += 1.0
                    rewards[0] -= 1.0
                else:
                    winner = -1
                    rewards[0] -= 2.0
                    rewards[1] -= 2.0
            return done, winner, alive0, alive1

        if self.step_idx >= self.max_steps:
            done = True
            if not self.any_attack:
                winner = -1
                rewards[0] -= 1.0
                rewards[1] -= 1.0
            else:
                if alive0 > alive1:
                    winner = 0
                    rewards[0] += 2.0
                    rewards[1] -= 2.0
                elif alive1 > alive0:
                    winner = 1
                    rewards[1] += 2.0
                    rewards[0] -= 2.0
                else:
                    hp0 = float(sum(max(0.0, hp) for hp in self.hps[0]))
                    hp1 = float(sum(max(0.0, hp) for hp in self.hps[1]))
                    if hp0 > hp1:
                        winner = 0
                        rewards[0] += 1.0
                        rewards[1] -= 1.0
                    elif hp1 > hp0:
                        winner = 1
                        rewards[1] += 1.0
                        rewards[0] -= 1.0
                    else:
                        winner = -1
            return done, winner, alive0, alive1

        return done, winner, alive0, alive1

    def _scale_rewards(self, rewards: List[float]) -> None:
        # 보상 스케일 정규화: 팩션별 초기 유닛 수로 나눠 학습 스케일을 맞춘다.
        scale0 = 1.0 / float(max(1, int(self.n_units[0])))
        scale1 = 1.0 / float(max(1, int(self.n_units[1])))
        rewards[0] = float(rewards[0]) * scale0
        rewards[1] = float(rewards[1]) * scale1

    def step(self, turn_action: Tuple[int, int, int]):
        """
        체스 룰: 교대 턴 + 턴당 1유닛만 행동.
        turn_action = (f, unit_index, action)
          - f: 현재 턴의 팩션(0 또는 1)이어야 함
          - unit_index: 해당 팩션 유닛 인덱스
          - action: 0~4 이동(정지/상/하/좌/우), 5 공격
        """
        rewards = [0.0, 0.0]

        f, i, a = self._sanitize_turn_action(turn_action)
        occupied = self._build_occupied_map()

        attacks_this_step = 0
        move_len = 0
        if self._is_alive_unit(f, i):
            move_len = self._apply_turn_move(f, i, a, occupied)

        # nearest는 "이동 이후"를 기준으로 한 번만 계산한다. (기존 구현과 동일한 타이밍)
        _, nearest_dist = self._nearest_enemy_all()

        # 공격: action >= move_pattern_len
        if self._is_alive_unit(f, i) and a >= int(move_len):
            attacks_this_step = self._apply_turn_attack(f, i, occupied, rewards)

        self._apply_distance_shaping(rewards, nearest_dist)

        # 무교전 조기 종료용 카운터
        if attacks_this_step == 0:
            self.no_attack_steps += 1
        else:
            self.no_attack_steps = 0

        self.step_idx += 1
        # 다음 턴으로
        self.side_to_act = 1 - int(getattr(self, "side_to_act", 0))

        done, winner, alive0, alive1 = self._resolve_terminal(rewards)

        info = {
            "winner": winner,
            "attack_distances": list(self.history["attack_distances"]),
            "alive": (alive0, alive1),
            "no_attack_steps": self.no_attack_steps,
            "side_to_act": int(getattr(self, "side_to_act", 0)),
        }

        self._scale_rewards(rewards)

        return self._get_obs(), rewards, done, info

