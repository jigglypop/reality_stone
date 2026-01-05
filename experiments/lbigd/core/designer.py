import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, List
from experiments.lbigd.core.simulation import run_simulation, run_simulation_all_pairs
from experiments.lbigd.core.metric import update as update_metric
from experiments.lbigd.core.lbo import laplacian_mul

try:
    import reality_stone as rs  # type: ignore
except Exception:
    rs = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

import concurrent.futures
import os

def wasserstein_distance_1d(u_values, v_values):
    """
    1차원 경험분포 간 2-Wasserstein 거리의 제곱(W2^2) 근사.

    1D에서 동일 가중치 표본(경험분포)은 정렬 후 평균 제곱차로 W2^2를 계산할 수 있습니다.
    (관련 설명: docs/blackbox.md, docs/evaluation.md)
    """
    u_sorted = np.sort(np.asarray(u_values, dtype=np.float64))
    v_sorted = np.sort(np.asarray(v_values, dtype=np.float64))
    
    if u_sorted.size == 0 or v_sorted.size == 0:
        return 0.0

    # 샘플 수가 다르면 보간법 사용해야 하나, 여기서는 편의상 샘플링으로 맞춤
    min_len = int(min(int(u_sorted.size), int(v_sorted.size)))
    if min_len <= 0:
        return 0.0
    # 균등 간격으로 샘플링하여 길이 맞춤
    u_indices = np.linspace(0, int(u_sorted.size) - 1, min_len).astype(np.int64)
    v_indices = np.linspace(0, int(v_sorted.size) - 1, min_len).astype(np.int64)
    
    u_resampled = u_sorted[u_indices]
    v_resampled = v_sorted[v_indices]
    
    diff = u_resampled - v_resampled
    return float(np.mean(diff * diff))


def _normal_quantiles(*, mean: float, std: float, n: int, clamp_min: float | None = None) -> np.ndarray:
    """
    재현 가능한 목표 분포 표본 생성:
    - 난수 샘플링 대신, (0,1) 균등 격자 quantile을 정규분포 inverse-CDF로 변환합니다.
    - SciPy 없이 torch.erfinv를 사용합니다.
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if std <= 0:
        raise ValueError("std must be > 0")

    with torch.no_grad():
        # (0,1)에서 0/1을 피하는 균등 격자: (i+0.5)/n
        p = (torch.arange(int(n), dtype=torch.float64) + 0.5) / float(n)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)  # Φ^{-1}(p)
        q = float(mean) + float(std) * z
        if clamp_min is not None:
            q = torch.clamp(q, min=float(clamp_min))
        return q.cpu().numpy().astype(np.float64, copy=False)

class DesignOptimizer:
    """
    blackbox.md 기반의 ES 최적화기
    목표: 승률 50:50 + 목표 교전 거리 분포
    """
    def __init__(
        self,
        initial_design: Dict[str, float],
        target_dist_mean: float,
        sigma: float = 0.1,
        lr: float = 0.1,
        n_samples: int = 4,
        seed_repeats: int = 1,
        train_episodes: int = 80,
        eval_episodes: int = 6,
        use_parallel: bool = True,
        max_workers: int = 0,
        base_seed: int = 42,
        verbose: bool = True,
        *,
        # objective weights (ultimate goal: win-rate 50:50)
        win_weight: float = 400.0,
        blowout_weight: float = 20.0,
        blowout_threshold: float = 0.95,
        draw_weight: float = 20.0,
        draw_weight_no_engagement: float = 200.0,
        w2_weight: float = 0.2,
        no_engagement_penalty: float = 100.0,
        lbo_loss_weight: float = 0.05,
        use_metric_lbo: bool = True,
        metric_tau: float = 0.35,
        metric_topk: int = 6,
        metric_lr: float = 0.25,
        metric_decay: float = 0.01,
        metric_w_max: float = 1.0,
        metric_beta: float = 0.15,
        metric_steps: int = 2,
        lbo_ema_decay: float = 0.97,
        sigma_int: float = 1.0,
        sigma_pattern: float = 2.0,
        use_torsion: bool = True,
        torsion_gamma: float = 0.05,
        torsion_steps: int = 1,
        torsion_seed: int | None = None,
    ):
        self.mean_design = initial_design.copy()
        self.target_dist_mean = target_dist_mean
        self.sigma = sigma # 탐색 노이즈 표준편차
        self.lr = lr       # 학습률
        self.n_samples = n_samples # ES 샘플 수 (짝수 권장)
        self.seed_repeats = int(max(1, int(seed_repeats)))
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.use_parallel = use_parallel
        self.base_seed = base_seed
        self.verbose = verbose
        self.win_weight = float(win_weight)
        self.blowout_weight = float(blowout_weight)
        self.blowout_threshold = float(blowout_threshold)
        self.draw_weight = float(draw_weight)
        self.draw_weight_no_engagement = float(draw_weight_no_engagement)
        self.w2_weight = float(w2_weight)
        self.no_engagement_penalty = float(no_engagement_penalty)
        self.lbo_loss_weight = float(lbo_loss_weight)
        self.use_metric_lbo = bool(use_metric_lbo)
        self.metric_tau = float(metric_tau)
        self.metric_topk = int(metric_topk)
        self.metric_lr = float(metric_lr)
        self.metric_decay = float(metric_decay)
        self.metric_w_max = float(metric_w_max)
        self.metric_beta = float(metric_beta)
        self.metric_steps = int(metric_steps)
        self.lbo_ema_decay = float(lbo_ema_decay)
        self._ema_lbo_win: float | None = None
        self._ema_lbo_loss: float | None = None
        self.sigma_int = float(sigma_int)
        self.sigma_pattern = float(sigma_pattern)
        self.use_torsion = bool(use_torsion)
        self.torsion_gamma = float(torsion_gamma)
        self.torsion_steps = int(torsion_steps)
        self.torsion_seed = torsion_seed

        # reality_stone (Rust) 기반 diffusion 엔진(있으면 사용):
        # - optimizable_keys 길이를 알기 전엔 만들 수 없으므로, 첫 smoothing 호출 시 lazy-init 한다.
        self._rs_diffusion = None
        self._rs_diffusion_dim: int | None = None
        self._rs_diffusion_dt: float | None = None

        if max_workers and max_workers > 0:
            self.max_workers = max_workers
        else:
            # 너무 과한 프로세스 생성 방지
            cpu = os.cpu_count() or 1
            self.max_workers = min(4, max(1, cpu // 2))
        
        # 최적화할 키 목록 (맵/유닛수/패턴/이동/사거리/스탯)
        self.optimizable_keys = [
            "width",
            "height",
            # 장애물(맵 기하)
            "obstacle_density",
            "obstacle_pattern",
            # 유닛 수 (킹은 고정 1개라 제외)
            "p0_unit0_units",
            "p0_unit1_units",
            "p0_unit2_units",
            "p0_unit3_units",
            "p0_unit4_units",
            "p1_unit0_units",
            "p1_unit1_units",
            "p1_unit2_units",
            "p1_unit3_units",
            "p1_unit4_units",
            # 이동 패턴 ID (0~11, ES가 탐색)
            "p0_unit0_pattern",
            "p0_unit1_pattern",
            "p0_unit2_pattern",
            "p0_unit3_pattern",
            "p0_unit4_pattern",
            "p1_unit0_pattern",
            "p1_unit1_pattern",
            "p1_unit2_pattern",
            "p1_unit3_pattern",
            "p1_unit4_pattern",
            # 공격 패턴 ID (0~12, ES가 탐색)
            "p0_unit0_attack_pattern",
            "p0_unit1_attack_pattern",
            "p0_unit2_attack_pattern",
            "p0_unit3_attack_pattern",
            "p0_unit4_attack_pattern",
            "p1_unit0_attack_pattern",
            "p1_unit1_attack_pattern",
            "p1_unit2_attack_pattern",
            "p1_unit3_attack_pattern",
            "p1_unit4_attack_pattern",
            # 이동거리
            "p0_unit0_move",
            "p0_unit1_move",
            "p0_unit2_move",
            "p1_unit0_move",
            "p1_unit1_move",
            "p1_unit2_move",
            "p1_unit4_move",
            # 사거리
            "p0_unit0_range",
            "p0_unit1_range",
            "p0_unit2_range",
            "p0_unit3_range",
            "p0_unit4_range",
            "p1_unit0_range",
            "p1_unit1_range",
            "p1_unit2_range",
            "p1_unit3_range",
            "p1_unit4_range",
            # 스탯
            "p0_unit0_hp",
            "p0_unit1_hp",
            "p0_unit0_damage",
            "p1_unit0_hp",
            "p1_unit1_hp",
            "p1_unit0_damage",
            "p1_unit1_damage",
            "p1_unit2_damage",
            "p1_unit3_hp",
            "p1_unit4_damage",
        ]

        # (옵션) 설계변수 간 "학습된 메트릭"을 만들고, 그 위에서 라플라시안으로 업데이트를 매끈하게 한다.
        # - 메트릭은 Hebbian-like rule(metric.update)로 갱신
        # - 업데이트 벡터(그라디언트)에 diffusion(-L g)을 걸어 "수로처럼" 안정된 방향으로 흐르게 함
        self._metric_w: np.ndarray | None = None
        self._metric_mask: np.ndarray | None = None
        self._torsion_sign: np.ndarray | None = None
        self._last_center_summary: Dict[str, float] | None = None
        if self.use_metric_lbo and len(self.optimizable_keys) > 2:
            self._metric_w, self._metric_mask, self._torsion_sign = self._init_metric_graph()

    def _ema_update(self, cur: float | None, x: float) -> float:
        if cur is None:
            return float(x)
        d = float(self.lbo_ema_decay)
        return d * float(cur) + (1.0 - d) * float(x)

    def _init_metric_graph(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        설계변수 키의 의미를 반영한 "구조 그래프"를 만든다.

        - ring처럼 임의 인덱스 이웃을 두지 않고,
          (i) 같은 유닛 슬롯 내부, (ii) 같은 속성의 양 진영 매칭, (iii) 맵/장애물 그룹
          위주로 엣지를 둔다.
        - 반환: (w_init, mask, torsion_sign)
          - w_init: (n,n) symmetric, diag=0
          - mask: (n,n) symmetric, diag=0, binary
          - torsion_sign: (n,n) skew-symmetric, diag=0, edge마다 +/-1 (비엣지는 0)
        """
        n = int(len(self.optimizable_keys))
        w = np.zeros((n, n), dtype=np.float32)
        idx = {k: i for i, k in enumerate(self.optimizable_keys)}

        def connect(a: str, b: str, weight: float = 1.0) -> None:
            ia = idx.get(a, None)
            ib = idx.get(b, None)
            if ia is None or ib is None or ia == ib:
                return
            ww = float(max(0.0, weight))
            if ww <= 0.0:
                return
            w[int(ia), int(ib)] = max(float(w[int(ia), int(ib)]), ww)
            w[int(ib), int(ia)] = max(float(w[int(ib), int(ia)]), ww)

        # 맵/장애물 그룹
        connect("width", "height", weight=1.0)
        connect("obstacle_density", "obstacle_pattern", weight=1.0)
        connect("width", "obstacle_density", weight=0.3)
        connect("height", "obstacle_density", weight=0.3)
        connect("width", "obstacle_pattern", weight=0.3)
        connect("height", "obstacle_pattern", weight=0.3)

        # 유닛 슬롯 내부(팩션별) 결합
        for prefix in ("p0", "p1"):
            for ui in range(5):
                base = f"{prefix}_unit{ui}"
                k_units = f"{base}_units"
                k_pat = f"{base}_pattern"
                k_atk = f"{base}_attack_pattern"
                k_move = f"{base}_move"
                k_range = f"{base}_range"
                k_hp = f"{base}_hp"
                k_dmg = f"{base}_damage"

                connect(k_units, k_pat, weight=1.0)
                connect(k_pat, k_atk, weight=1.0)
                connect(k_move, k_range, weight=0.8)
                connect(k_hp, k_dmg, weight=0.6)

                # "물량(유닛수)"가 플레이 패턴에 영향을 주는 축으로도 묶어준다.
                connect(k_units, k_move, weight=0.4)
                connect(k_units, k_range, weight=0.4)
                connect(k_units, k_hp, weight=0.4)
                connect(k_units, k_dmg, weight=0.4)

        # 진영 간 매칭(동일 슬롯/동일 속성은 약결합)
        for ui in range(5):
            for suffix in ("units", "pattern", "attack_pattern", "move", "range", "hp", "damage"):
                connect(f"p0_unit{ui}_{suffix}", f"p1_unit{ui}_{suffix}", weight=0.5)

        # 고립 노드 방지: degree=0이면 width에 약결합(없으면 첫 키에 연결)
        deg = w.sum(axis=1)
        anchor = "width" if "width" in idx else self.optimizable_keys[0]
        for i, d in enumerate(deg.tolist()):
            if float(d) <= 0.0:
                connect(self.optimizable_keys[int(i)], anchor, weight=0.2)

        # mask는 "허용 엣지"이고, update_metric에서 affinity를 제한하는 데 쓴다.
        mask = (w > 0.0).astype(np.float32)
        np.fill_diagonal(mask, 0.0)
        np.fill_diagonal(w, 0.0)

        # torsion_sign: 엣지마다 방향(부호)을 하나 정해, skew-symmetric 행렬을 만든다.
        # - 이 행렬은 "회전/curl 성분"을 만들어내는 최소 토션 항으로 사용한다.
        seed = int(self.torsion_seed) if self.torsion_seed is not None else int(self.base_seed)
        rng = np.random.default_rng(int(seed))
        tors = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if float(mask[i, j]) <= 0.0:
                    continue
                s = -1.0 if bool(rng.integers(0, 2)) else 1.0
                tors[i, j] = float(s)
                tors[j, i] = float(-s)
        np.fill_diagonal(tors, 0.0)
        return w.astype(np.float32), mask.astype(np.float32), tors.astype(np.float32)

    def _key_sigma(self, key: str) -> float:
        """
        이산/정수 변수는 작은 sigma에서 거의 변하지 않아 탐색이 붕괴하기 쉬워서,
        key 타입에 따라 sigma를 다르게 둔다.
        """
        k = str(key)
        if k in ("width", "height"):
            return float(max(1e-6, self.sigma_int))
        if k == "obstacle_pattern":
            return float(max(1e-6, self.sigma_pattern))
        if k.endswith("_units"):
            return float(max(1e-6, self.sigma_int))
        if k.endswith("_attack_pattern") or k.endswith("_pattern"):
            return float(max(1e-6, self.sigma_pattern))
        return float(max(1e-6, self.sigma))

    def get_loss(self, stats: Dict) -> float:
        # 궁극 목표: 50:50 밸런스 (승률 기반). draw는 별도 페널티로 다룬다.
        n_factions = int(stats.get("n_factions", 2))

        dist_samples = stats.get("distance_samples", [])
        engaged = len(dist_samples) > 0
        
        if n_factions > 2 and "win_matrix" in stats:
            # 다팩션: 모든 쌍의 승률 균형
            win_matrix = stats["win_matrix"]
            win_diff_sum = 0.0
            blowout_count = 0
            n_pairs = 0
            
            for (i, j), win_rate in win_matrix.items():
                if i < j:  # 중복 방지
                    p_ij = float(win_rate)
                    p_ji = float(win_matrix.get((j, i), 0.0))
                    decisive = float(p_ij + p_ji)
                    p_decisive = (p_ij / decisive) if decisive > 1e-8 else 0.5
                    win_diff_sum += (p_decisive - 0.5) ** 2
                    if p_decisive <= (1.0 - float(self.blowout_threshold)) or p_decisive >= float(self.blowout_threshold):
                        blowout_count += 1
                    n_pairs += 1
            
            win_diff = win_diff_sum / max(1, n_pairs)
            blowout_penalty = float(self.blowout_weight) * float(blowout_count)
        else:
            # 2팩션: draw를 제외한 "결정적 승부" 비율로 50:50을 맞춘다.
            p0 = float(stats.get("p0_win_rate", 0.5))
            p1 = float(stats.get("p1_win_rate", 0.5))
            decisive = float(p0 + p1)
            p0_decisive = (p0 / decisive) if decisive > 1e-8 else 0.5
            win_diff = (p0_decisive - 0.5) ** 2
            blowout_penalty = 0.0
            if p0_decisive <= (1.0 - float(self.blowout_threshold)) or p0_decisive >= float(self.blowout_threshold):
                blowout_penalty = float(self.blowout_weight)
        
        # 2. 분포 정합 손실: Wasserstein 거리
        if dist_samples:
            target_samples = _normal_quantiles(mean=float(self.target_dist_mean), std=1.0, n=len(dist_samples), clamp_min=0.0)
            w2_dist = wasserstein_distance_1d(dist_samples, target_samples)
        else:
            w2_dist = 0.0

        # 2-b. 교전이 아예 없으면 강한 페널티
        no_engagement_penalty = 0.0 if engaged else float(self.no_engagement_penalty)
        
        # 3. 무승부 페널티
        draw_rate = float(stats.get("draw_rate", 0.0))
        dw = float(self.draw_weight) if engaged else float(self.draw_weight_no_engagement)
        draw_penalty = draw_rate * dw
        
        # 총 손실
        total_loss = float(self.win_weight) * float(win_diff) + blowout_penalty + float(self.w2_weight) * float(w2_dist) + draw_penalty + no_engagement_penalty
        return total_loss

    def get_harmonic_loss(self, stats_pos: Dict, stats_neg: Dict) -> float:
        """
        라플라스-벨트라미(Laplace-Beltrami) 정규화:
        승률 지형의 곡률(Curvature)을 페널티로 부과합니다.
        P(x+e) + P(x-e) ~ 1.0 (Harmonic condition at P=0.5)
        """
        p0_pos = float(stats_pos.get("p0_win_rate", 0.5))
        p0_neg = float(stats_neg.get("p0_win_rate", 0.5))
        
        # 1.0에서 벗어난 정도가 곧 Laplacian의 크기 (2차 미분 근사)
        curvature = abs((p0_pos + p0_neg) - 1.0)
        return curvature * 10.0

    def get_lbo_curvature(self, stats_center: Dict, stats_pos: Dict, stats_neg: Dict, *, denom: float) -> float:
        """
        설계공간에서의 LBO(라플라시안) 근사:
          ΔP(x) ≈ (P(x+Δ) + P(x-Δ) - 2P(x)) / ||Δ||^2

        - stats_*: 동일 seed(=CRN)로 평가된 통계
        - denom: ||Δ||^2 (실제 step 벡터의 제곱노름; per-key sigma 반영)
        """
        denom = float(max(1e-6, float(denom)))

        n_factions = int(stats_center.get("n_factions", 2))
        if n_factions > 2 and "win_matrix" in stats_center:
            win_c = stats_center["win_matrix"]
            win_p = stats_pos["win_matrix"]
            win_n = stats_neg["win_matrix"]
            laps = []
            for (i, j), p_c in win_c.items():
                if i < j:
                    p_p = float(win_p.get((i, j), p_c))
                    p_n = float(win_n.get((i, j), p_c))
                    lap = (p_p + p_n - 2.0 * float(p_c)) / denom
                    laps.append(lap)
            if not laps:
                return 0.0
            return float(np.mean(np.abs(np.array(laps, dtype=np.float64))))

        p_c = float(stats_center.get("p0_win_rate", 0.5))
        p_p = float(stats_pos.get("p0_win_rate", p_c))
        p_n = float(stats_neg.get("p0_win_rate", p_c))
        lap = (p_p + p_n - 2.0 * p_c) / denom
        return float(abs(lap))

    def _clamp_design(self, d: Dict[str, float]) -> Dict[str, float]:
        """
        ES는 연속값을 뱉으므로, 격자/유닛수/스탯에 대해 최소한의 정수화/클램프를 적용합니다.
        """
        out = dict(d)

        # 맵 크기 (체스보다 크게 유지)
        w = int(round(float(out.get("width", 12))))
        h = int(round(float(out.get("height", 12))))
        w = max(10, min(24, w))
        h = max(10, min(24, h))
        # 대칭 배치/장애물 미러링을 단순하게 유지하기 위해 짝수 보드 우선
        if w % 2 == 1:
            w = w + 1 if w < 24 else w - 1
        if h % 2 == 1:
            h = h + 1 if h < 24 else h - 1
        out["width"] = int(w)
        out["height"] = int(h)

        # 장애물(밀도/패턴)
        out["obstacle_density"] = float(max(0.0, min(0.35, float(out.get("obstacle_density", 0.0)))))
        out["obstacle_pattern"] = int(max(0, min(4, int(round(float(out.get("obstacle_pattern", 0)))))))

        # 유닛 수(타입별): unit0~unit4, 0도 허용 (총합 변동 가능)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_units"
                val = int(round(float(out.get(key, 0))))
                if prefix == "p0":
                    out[key] = max(0, min(8, val))
                else:
                    out[key] = max(0, min(5, val))

        # 킹만 남는 구성을 막기 위해(퇴화 방지), 최소 1개는 유지
        for prefix in ("p0", "p1"):
            keys = [f"{prefix}_unit{i}_units" for i in range(5)]
            total = int(sum(int(out.get(k, 0)) for k in keys))
            if total <= 0:
                out[keys[0]] = 1
        
        # 이동 패턴 ID (0~11 정수)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_pattern"
                val = int(round(float(out.get(key, 0))))
                out[key] = max(0, min(11, val))

        # 공격 패턴 ID (0~12 정수). 기본은 이동 패턴을 따르도록 한다.
        for prefix in ("p0", "p1"):
            for i in range(5):
                move_key = f"{prefix}_unit{i}_pattern"
                atk_key = f"{prefix}_unit{i}_attack_pattern"
                if atk_key not in out:
                    out[atk_key] = int(out.get(move_key, 0))
                val = int(round(float(out.get(atk_key, 0))))
                out[atk_key] = max(0, min(12, val))

        # 이동 거리 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_move"
                if key in out:
                    out[key] = float(max(1.0, min(5.0, float(out.get(key, 1.0)))))
        
        # 사거리 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_range"
                if key in out:
                    out[key] = float(max(1.0, min(8.0, float(out.get(key, 1.0)))))
        
        # HP (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_hp"
                if key in out:
                    out[key] = float(max(1.0, min(10.0, float(out.get(key, 2.0)))))
        
        # 데미지 (unit0~unit4)
        for prefix in ("p0", "p1"):
            for i in range(5):
                key = f"{prefix}_unit{i}_damage"
                if key in out:
                    out[key] = float(max(0.5, min(3.0, float(out.get(key, 1.0)))))

        # 에피소드 길이(너무 길면 속도 폭발)
        out["max_steps"] = int(max(60, min(200, int(round(float(out.get("max_steps", 120)))))))

        return out

    def _smooth_gradients_with_metric(self, avg_grad: Dict[str, float]) -> Dict[str, float]:
        """
        학습된 메트릭 그래프 위에서 ES 그라디언트를 diffusion으로 매끈하게 만든다.

        - g <- g - beta * L g  (L = D - W)
        - W는 매 step의 g로부터 Hebbian-like 업데이트로 갱신한다.
        """
        if not self.use_metric_lbo:
            return avg_grad
        if self._metric_w is None:
            return avg_grad
        if self._metric_mask is None:
            return avg_grad
        if self.metric_steps <= 0:
            return avg_grad

        g = np.array([float(avg_grad[k]) for k in self.optimizable_keys], dtype=np.float32)
        # 메트릭 학습은 scale에 민감하므로, magnitude로 1차 정규화한다.
        scale = float(np.mean(np.abs(g))) if g.size > 0 else 0.0
        g_for_metric = (g / float(scale + 1e-8)).astype(np.float32, copy=False) if scale > 0 else g

        self._metric_w = update_metric(
            self._metric_w,
            g_for_metric,
            lr=float(self.metric_lr),
            tau=float(self.metric_tau),
            topk=int(self.metric_topk),
            decay=float(self.metric_decay),
            w_max=float(self.metric_w_max),
            mask=self._metric_mask,
        )

        def torsion_cayley_step(vec: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
            """
            Cayley transform 기반 토션 스텝 (노름 보존 성질):
              vec <- (I - aR)^{-1} (I + aR) vec,  a=gamma/2

            R가 skew-symmetric면 위 변환은 (이상적으로) 직교 변환이며 ||vec||를 보존한다.
            """
            g = float(gamma)
            if g == 0.0:
                return vec
            # 수치 안전을 위해 skew-symmetric 강제
            r = 0.5 * (r - r.T)
            n = int(r.shape[0])
            if n <= 0:
                return vec

            a = 0.5 * g
            r64 = r.astype(np.float64, copy=False)
            v64 = vec.astype(np.float64, copy=False)
            i64 = np.eye(n, dtype=np.float64)
            lhs = i64 - a * r64
            rhs = (i64 + a * r64) @ v64
            out = np.linalg.solve(lhs, rhs)
            return out.astype(np.float32, copy=False)

        g_smooth = g

        # (옵션) rs diffusion lazy-init
        if self._rs_diffusion is None and rs is not None:
            engine_cls = getattr(rs, "PyRiemannianDiffusion", None)
            if engine_cls is not None:
                try:
                    self._rs_diffusion = engine_cls(
                        int(g_smooth.size),
                        0.0,  # alpha=0 => flow - h를 그대로 반영
                        float(self.metric_beta),  # dt를 metric_beta로 매핑
                    )
                    self._rs_diffusion_dim = int(g_smooth.size)
                    self._rs_diffusion_dt = float(self.metric_beta)
                except Exception:
                    self._rs_diffusion = None
                    self._rs_diffusion_dim = None
                    self._rs_diffusion_dt = None
        elif self._rs_diffusion is not None:
            # dt/차원 변경 시 재생성
            if self._rs_diffusion_dim != int(g_smooth.size) or (self._rs_diffusion_dt is not None and abs(float(self._rs_diffusion_dt) - float(self.metric_beta)) > 1e-12):
                self._rs_diffusion = None
                self._rs_diffusion_dim = None
                self._rs_diffusion_dt = None
                if rs is not None:
                    engine_cls = getattr(rs, "PyRiemannianDiffusion", None)
                    if engine_cls is not None:
                        try:
                            self._rs_diffusion = engine_cls(int(g_smooth.size), 0.0, float(self.metric_beta))
                            self._rs_diffusion_dim = int(g_smooth.size)
                            self._rs_diffusion_dt = float(self.metric_beta)
                        except Exception:
                            self._rs_diffusion = None
                            self._rs_diffusion_dim = None
                            self._rs_diffusion_dt = None

        torsion_enabled = bool(self.use_torsion) and (self._torsion_sign is not None) and int(self.torsion_steps) > 0
        for _ in range(int(self.metric_steps)):
            # diffusion / curvature smoothing (Laplacian)
            # - 기본: g <- g - beta * L g
            # - rs가 있으면: flow = g - Lg 를 만들어 RiemannianDiffusion(step_cpu)로 동일 스텝을 수행(핵심 연산을 rs로 연결)
            if self._rs_diffusion is not None:
                lap = laplacian_mul(self._metric_w, g_smooth.astype(np.float32, copy=False))
                h = g_smooth.astype(np.float32, copy=False)[None, :]
                flow = (g_smooth - lap).astype(np.float32, copy=False)[None, :]
                out = self._rs_diffusion.step_cpu(h, flow)
                g_smooth = np.asarray(out, dtype=np.float32)[0]
            else:
                g_smooth = g_smooth - float(self.metric_beta) * laplacian_mul(self._metric_w, g_smooth)

            # torsion / curl-like component (skew-symmetric):
            #   R = (torsion_sign ∘ W) (W symmetric, torsion_sign skew-symmetric => R skew-symmetric)
            #   g <- Cayley(R, gamma) g  (회전 성분을 안정적으로 반영)
            if torsion_enabled:
                r = (self._metric_w * self._torsion_sign).astype(np.float32, copy=False)
                gamma = float(self.torsion_gamma)
                for _ in range(int(self.torsion_steps)):
                    g_smooth = torsion_cayley_step(g_smooth, r, gamma=float(gamma))
        return {k: float(g_smooth[i]) for i, k in enumerate(self.optimizable_keys)}

    def _eval_pair(self, design_pos: Dict[str, float], design_neg: Dict[str, float], seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 다팩션이면 all_pairs, 아니면 기존 방식
        n_factions = int(design_pos.get("n_factions", 2))
        if n_factions > 2:
            stats_pos = run_simulation_all_pairs(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_neg = run_simulation_all_pairs(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        else:
            stats_pos = run_simulation(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
            stats_neg = run_simulation(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed)
        return stats_pos, stats_neg

    def _eval_triplet(
        self,
        design_center: Dict[str, float],
        design_pos: Dict[str, float],
        design_neg: Dict[str, float],
        seed: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        동일 seed(CRN)에서 center/pos/neg를 함께 평가하여, 설계공간 LBO(2차차분)를 계산할 수 있게 한다.
        """
        repeats = int(max(1, int(getattr(self, "seed_repeats", 1))))
        n_factions = int(design_center.get("n_factions", 2))

        def merge_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not stats_list:
                return {}
            if len(stats_list) == 1:
                return stats_list[0]

            # distance samples: concat, then recompute mean/std
            all_dist: List[float] = []
            for s in stats_list:
                all_dist.extend(list(s.get("distance_samples", []) or []))

            if n_factions > 2 and all("win_matrix" in s for s in stats_list):
                acc: Dict[Tuple[int, int], float] = {}
                cnt: Dict[Tuple[int, int], int] = {}
                for s in stats_list:
                    wm = s.get("win_matrix", {}) or {}
                    for k, v in wm.items():
                        kk = (int(k[0]), int(k[1]))
                        acc[kk] = float(acc.get(kk, 0.0)) + float(v)
                        cnt[kk] = int(cnt.get(kk, 0)) + 1
                win_matrix = {k: float(acc[k]) / float(max(1, cnt.get(k, 0))) for k in acc.keys()}
                draw_rate = float(np.mean([float(s.get("draw_rate", 0.0)) for s in stats_list]))
                avg_distance = float(np.mean(np.asarray(all_dist, dtype=np.float64))) if all_dist else 0.0
                distance_std = float(np.std(np.asarray(all_dist, dtype=np.float64))) if all_dist else 0.0
                out = {
                    "n_factions": int(n_factions),
                    "win_matrix": win_matrix,
                    "draw_rate": draw_rate,
                    "avg_distance": avg_distance,
                    "distance_std": distance_std,
                    "distance_samples": all_dist,
                }
                out["p0_win_rate"] = float(win_matrix.get((0, 1), 0.5))
                out["p1_win_rate"] = float(win_matrix.get((1, 0), 0.5))
                return out

            # 2-faction (or fallback): average scalar stats
            p0 = float(np.mean([float(s.get("p0_win_rate", 0.5)) for s in stats_list]))
            p1 = float(np.mean([float(s.get("p1_win_rate", 0.5)) for s in stats_list]))
            draw_rate = float(np.mean([float(s.get("draw_rate", 0.0)) for s in stats_list]))
            avg_distance = float(np.mean(np.asarray(all_dist, dtype=np.float64))) if all_dist else 0.0
            distance_std = float(np.std(np.asarray(all_dist, dtype=np.float64))) if all_dist else 0.0
            out = dict(stats_list[0])
            out.update(
                {
                    "p0_win_rate": p0,
                    "p1_win_rate": p1,
                    "draw_rate": draw_rate,
                    "avg_distance": avg_distance,
                    "distance_std": distance_std,
                    "distance_samples": all_dist,
                }
            )
            return out

        stats_cs: List[Dict[str, Any]] = []
        stats_ps: List[Dict[str, Any]] = []
        stats_ns: List[Dict[str, Any]] = []
        for r in range(repeats):
            # CRN: repeat마다 base seed만 바꾸고, center/pos/neg는 동일 seed를 공유한다.
            seed_r = int(seed + r * 100_000)
            if n_factions > 2:
                stats_cs.append(
                    run_simulation_all_pairs(design_center, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r)
                )
                stats_ps.append(
                    run_simulation_all_pairs(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r)
                )
                stats_ns.append(
                    run_simulation_all_pairs(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r)
                )
            else:
                stats_cs.append(run_simulation(design_center, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r))
                stats_ps.append(run_simulation(design_pos, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r))
                stats_ns.append(run_simulation(design_neg, train_episodes=self.train_episodes, eval_episodes=self.eval_episodes, seed=seed_r))

        return merge_stats(stats_cs), merge_stats(stats_ps), merge_stats(stats_ns)

    def _sample_antithetic_pairs(self, step_index: int) -> Tuple[List[Tuple[int, Dict[str, float], Dict[str, float], int]], List[Dict[str, float]]]:
        """
        Antithetic sampling(+/-)을 구성합니다.
        반환:
          - pairs: (sample_index, design_pos, design_neg, seed)
          - epsilons: epsilon 벡터(dict) 목록 (index로 접근)
        """
        pairs: List[Tuple[int, Dict[str, float], Dict[str, float], int]] = []
        epsilons: List[Dict[str, float]] = []

        # NOTE: simulation 내부에서 np.random.seed를 건드리므로, ES의 epsilon은 로컬 RNG로 고정해 재현성을 확보한다.
        rng = np.random.default_rng(int(self.base_seed + step_index * 10_000 + 7))
        half = int(self.n_samples // 2)
        for i in range(half):
            epsilon = {k: float(rng.standard_normal()) for k in self.optimizable_keys}
            epsilons.append(epsilon)

            design_pos = self.mean_design.copy()
            design_neg = self.mean_design.copy()
            for k in self.optimizable_keys:
                base = float(self.mean_design.get(k, 0.0))
                sigma_k = float(self._key_sigma(k))
                design_pos[k] = base + sigma_k * float(epsilon[k])
                design_neg[k] = base - sigma_k * float(epsilon[k])

            design_pos = self._clamp_design(design_pos)
            design_neg = self._clamp_design(design_neg)

            seed = int(self.base_seed + step_index * 1000 + i)
            pairs.append((i, design_pos, design_neg, seed))

        return pairs, epsilons

    def _evaluate_triplets(self, pairs: List[Tuple[int, Dict[str, float], Dict[str, float], int]]):
        """
        center/pos/neg를 같은 seed(CRN)로 평가합니다.
        반환: (sample_index, stats_center, stats_pos, stats_neg) 목록(정렬됨)
        """
        if not pairs:
            return []

        # tqdm: pair 단위로 진행률 표시 (없으면 no-op)
        if tqdm is None:
            class _NoTqdm:
                def update(self, n: int = 1) -> None:
                    return None

                def close(self) -> None:
                    return None

            pbar = _NoTqdm()
        else:
            # quiet 모드에서는 tqdm 출력 비활성화
            pbar = tqdm(total=len(pairs), desc="ES Eval", leave=False, disable=not bool(self.verbose))

        def _serial_eval():
            out = []
            for i, dpos, dneg, seed in pairs:
                stats_c, stats_pos, stats_neg = self._eval_triplet(self.mean_design, dpos, dneg, seed)
                out.append((i, stats_c, stats_pos, stats_neg))
                pbar.update(1)
            return out

        n_factions = int(self.mean_design.get("n_factions", 2))
        sim_func = run_simulation_all_pairs if n_factions > 2 else run_simulation

        eval_results = []
        if self.use_parallel and self.max_workers > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = {
                        ex.submit(sim_func, self.mean_design, self.train_episodes, self.eval_episodes, seed): (i, "center")
                        for i, _, _, seed in pairs
                    }
                    futures.update(
                        {
                            ex.submit(sim_func, dpos, self.train_episodes, self.eval_episodes, seed): (i, "pos")
                            for i, dpos, _, seed in pairs
                        }
                    )
                    futures.update(
                        {
                            ex.submit(sim_func, dneg, self.train_episodes, self.eval_episodes, seed): (i, "neg")
                            for i, _, dneg, seed in pairs
                        }
                    )

                    tmp: Dict[int, Dict[str, Any]] = {}
                    done_pair = {i: 0 for i, _, _, _ in pairs}
                    for fut in concurrent.futures.as_completed(futures):
                        i, sign = futures[fut]
                        stats = fut.result()
                        if i not in tmp:
                            tmp[i] = {}
                        tmp[i][sign] = stats
                        done_pair[i] += 1
                        if done_pair[i] == 3:
                            eval_results.append((i, tmp[i]["center"], tmp[i]["pos"], tmp[i]["neg"]))
                            pbar.update(1)
            except Exception:
                eval_results = _serial_eval()
        else:
            eval_results = _serial_eval()

        pbar.close()
        eval_results.sort(key=lambda x: x[0])
        return eval_results

    def _accumulate_es_gradients(
        self,
        eval_results: list,
        epsilons: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], list]:
        """
        평가 결과를 ES 그라디언트 누적과 로깅용 결과로 변환합니다.
        반환:
          - gradients: optimizable_keys별 누적 그라디언트(sum)
          - results: (loss_pos, loss_neg, lbo) 목록
        """
        gradients = {k: 0.0 for k in self.optimizable_keys}
        results = []

        for i, stats_center, stats_pos, stats_neg in eval_results:
            epsilon = epsilons[int(i)]
            step_norm2 = 0.0
            for k in self.optimizable_keys:
                sigma_k = float(self._key_sigma(k))
                e = float(epsilon[k])
                step_norm2 += (sigma_k * sigma_k) * (e * e)

            loss_center = self.get_loss(stats_center)
            loss_pos = self.get_loss(stats_pos)
            loss_neg = self.get_loss(stats_neg)

            denom = float(max(1e-6, step_norm2))
            lbo_win = self.get_lbo_curvature(stats_center, stats_pos, stats_neg, denom=denom)
            lbo_loss = float(abs((float(loss_pos) + float(loss_neg) - 2.0 * float(loss_center)) / denom))

            # 스케일 자동 정규화(노이즈/환경에 따라 곡률의 절대 스케일이 크게 달라질 수 있음)
            self._ema_lbo_win = self._ema_update(self._ema_lbo_win, float(lbo_win))
            self._ema_lbo_loss = self._ema_update(self._ema_lbo_loss, float(lbo_loss))
            win_norm = float(lbo_win) / float((self._ema_lbo_win or 0.0) + 1e-8)
            loss_norm = float(lbo_loss) / float((self._ema_lbo_loss or 0.0) + 1e-8)
            lbo_total = float(win_norm) + float(self.lbo_loss_weight) * float(loss_norm)
            lbo_w = 1.0 / (1.0 + float(lbo_total))

            if self.verbose:
                print(
                    f"    Sample {i+1} (+): Loss={loss_pos:.4f} LBOw={lbo_win:.4f} LBOL={lbo_loss:.4f} w={lbo_w:.3f} | P0={stats_pos['p0_win_rate']:.2f} "
                    f"P1={stats_pos['p1_win_rate']:.2f} Draw={stats_pos['draw_rate']:.2f} Dist={stats_pos['avg_distance']:.2f}"
                )
                print(
                    f"    Sample {i+1} (-): Loss={loss_neg:.4f} LBOw={lbo_win:.4f} LBOL={lbo_loss:.4f} w={lbo_w:.3f} | P0={stats_neg['p0_win_rate']:.2f} "
                    f"P1={stats_neg['p1_win_rate']:.2f} Draw={stats_neg['draw_rate']:.2f} Dist={stats_neg['avg_distance']:.2f}"
                )

            diff = float(loss_pos) - float(loss_neg)
            for k in self.optimizable_keys:
                sigma_k = float(self._key_sigma(k))
                gradients[k] += float(lbo_w) * diff * float(epsilon[k]) * (1.0 / float(2.0 * sigma_k))

            results.append((loss_pos, loss_neg, lbo_win, lbo_loss))

        return gradients, results

    def step(self, step_index: int = 0):
        # ES (Score Function Estimator)
        # J_sigma(x) ~ E[J(x + sigma*epsilon)]
        pairs, epsilons = self._sample_antithetic_pairs(step_index=step_index)
        eval_results = self._evaluate_triplets(pairs)

        # 현재(mean) 설계의 "성능"을 같은 프로토콜에서 요약해 둔다.
        try:
            centers = [stats_c for _, stats_c, _, _ in eval_results]
            if centers:
                p0 = float(np.mean([float(s.get("p0_win_rate", 0.5)) for s in centers]))
                p1 = float(np.mean([float(s.get("p1_win_rate", 0.5)) for s in centers]))
                decisive = float(p0 + p1)
                p0_dec = float(p0 / decisive) if decisive > 1e-8 else 0.5
                draw = float(np.mean([float(s.get("draw_rate", 0.0)) for s in centers]))
                dist = float(np.mean([float(s.get("avg_distance", 0.0)) for s in centers]))
                self._last_center_summary = {"p0": p0, "p1": p1, "p0_dec": p0_dec, "draw": draw, "dist": dist}
            else:
                self._last_center_summary = None
        except Exception:
            self._last_center_summary = None

        gradients, results = self._accumulate_es_gradients(eval_results, epsilons)

        # 평균 그라디언트로 업데이트
        denom = float(max(1, int(self.n_samples // 2)))
        avg_grad = {k: float(v) / denom for k, v in gradients.items()}

        avg_grad = self._smooth_gradients_with_metric(avg_grad)

        for k in self.optimizable_keys:
            base = float(self.mean_design.get(k, 0.0))
            self.mean_design[k] = base - float(self.lr) * float(avg_grad[k])

        # mean 자체도 클램프 (폭주/0으로 붕괴 방지)
        self.mean_design = self._clamp_design(self.mean_design)
            
        # 로깅용 스칼라: (+/-) 평균 손실
        avg_loss = float(np.mean([(float(r[0]) + float(r[1])) * 0.5 for r in results])) if results else 0.0
        return avg_loss, self.mean_design


def _default_initial_design(*, seed: int = 42) -> Dict[str, float]:
    # 최소 기동용 기본 설계 (부족한 키는 clamp/환경 기본값으로 보정됨)
    return {
        "width": 12,
        "height": 12,
        "obstacle_density": 0.0,
        "obstacle_pattern": 0,
        "max_steps": 120,
        "no_attack_limit": 20,
        "shaping_scale": 0.02,
        "n_factions": 2,
        "seed": int(seed),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="lbigd")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-dist-mean", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seed-repeats", type=int, default=1)
    p.add_argument("--train-episodes", type=int, default=80)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--no-parallel", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-torsion", action="store_true")
    p.add_argument("--fixed-budget", action="store_true")
    args = p.parse_args(argv)

    init = _default_initial_design(seed=int(args.seed))
    opt = DesignOptimizer(
        initial_design=init,
        target_dist_mean=float(args.target_dist_mean),
        sigma=float(args.sigma),
        lr=float(args.lr),
        n_samples=int(args.n_samples),
        seed_repeats=int(args.seed_repeats),
        train_episodes=int(args.train_episodes),
        eval_episodes=int(args.eval_episodes),
        use_parallel=not bool(args.no_parallel),
        base_seed=int(args.seed),
        verbose=not bool(args.quiet),
        use_torsion=not bool(args.no_torsion),
    )

    if not bool(args.quiet):
        if rs is not None:
            print(f"[rs] rust_ext={getattr(rs, '_has_rust_ext', False)} cuda={getattr(rs, '_has_cuda', False)}")
        if bool(args.fixed_budget):
            print(
                f"[start] steps={int(args.steps)} n_samples={int(args.n_samples)} "
                f"train={int(args.train_episodes)} eval={int(args.eval_episodes)} repeats={int(args.seed_repeats)} (fixed)"
            )
        else:
            # late-stage: eval > train (balance estimation focus)
            train_start = int(args.train_episodes)
            train_end = min(train_start, max(10, train_start // 4))
            eval_start = int(args.eval_episodes)
            eval_end = max(eval_start, int(train_end + 1))
            rep_start = int(args.seed_repeats)
            rep_end = max(rep_start, 3)
            print(
                f"[start] steps={int(args.steps)} n_samples={int(args.n_samples)} "
                f"train={train_start}->{train_end} eval={eval_start}->{eval_end} repeats={rep_start}->{rep_end} (anneal)"
            )

    for step in range(int(args.steps)):
        # 후반으로 갈수록 "교전(평가) > 학습(훈련)" 비중으로 전환한다.
        if not bool(args.fixed_budget) and int(args.steps) > 1:
            t = float(step) / float(max(1, int(args.steps) - 1))
            train_start = int(args.train_episodes)
            train_end = min(train_start, max(10, train_start // 4))
            eval_start = int(args.eval_episodes)
            eval_end = max(eval_start, int(train_end + 1))
            rep_start = int(args.seed_repeats)
            rep_end = max(rep_start, 3)

            train_cur = int(round(train_start + t * float(train_end - train_start)))
            eval_cur = int(round(eval_start + t * float(eval_end - eval_start)))
            rep_cur = int(round(rep_start + t * float(rep_end - rep_start)))

            opt.train_episodes = int(max(1, train_cur))
            opt.eval_episodes = int(max(1, eval_cur))
            opt.seed_repeats = int(max(1, rep_cur))

        loss, design = opt.step(step)
        if not bool(args.quiet):
            s = getattr(opt, "_last_center_summary", None)
            if isinstance(s, dict) and s:
                print(
                    f"[step {step:04d}] loss={loss:.4f} "
                    f"train={int(getattr(opt, 'train_episodes', 0))} eval={int(getattr(opt, 'eval_episodes', 0))} repeats={int(getattr(opt, 'seed_repeats', 0))} "
                    f"center(p0_dec={float(s.get('p0_dec', 0.5)):.3f} draw={float(s.get('draw', 0.0)):.2f} dist={float(s.get('dist', 0.0)):.2f}) "
                    f"width={design.get('width')} height={design.get('height')}"
                )
            else:
                print(
                    f"[step {step:04d}] loss={loss:.4f} "
                    f"train={int(getattr(opt, 'train_episodes', 0))} eval={int(getattr(opt, 'eval_episodes', 0))} repeats={int(getattr(opt, 'seed_repeats', 0))} "
                    f"width={design.get('width')} height={design.get('height')}"
                )

    if not bool(args.quiet):
        print("[done] final_design:")
        for k in sorted(design.keys()):
            print(f"  {k}: {design[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

