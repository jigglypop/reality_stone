"""
Transformer -> RS-ULF 변환기 (Rust 바인딩 사용)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List

from tqdm import tqdm

import reality_stone as rs


def extract_transformer_layer_weights(model, layer_idx: int) -> Dict[str, torch.Tensor]:
    try:
        layer = model.model.layers[layer_idx]
    except AttributeError:
        try:
            layer = model.transformer.h[layer_idx]
        except AttributeError:
            layer = model.layers[layer_idx]
    
    weights = {}
    
    try:
        weights['WQ'] = layer.self_attn.q_proj.weight.detach()
        weights['WK'] = layer.self_attn.k_proj.weight.detach()
        weights['WV'] = layer.self_attn.v_proj.weight.detach()
        weights['WO'] = layer.self_attn.o_proj.weight.detach()
    except AttributeError:
        if hasattr(layer, 'attn'):
            qkv = layer.attn.c_attn.weight.detach()
            d = qkv.size(0) // 3
            weights['WQ'] = qkv[:d, :]
            weights['WK'] = qkv[d:2*d, :]
            weights['WV'] = qkv[2*d:, :]
            weights['WO'] = layer.attn.c_proj.weight.detach()
    
    try:
        weights['W1'] = layer.mlp.gate_proj.weight.detach()
        weights['W2'] = layer.mlp.down_proj.weight.detach()
        if hasattr(layer.mlp, 'up_proj'):
            weights['W_up'] = layer.mlp.up_proj.weight.detach()
        else:
            weights['W_up'] = None
    except AttributeError:
        weights['W1'] = layer.mlp.c_fc.weight.detach()
        weights['W2'] = layer.mlp.c_proj.weight.detach()
        weights['W_up'] = None
    
    try:
        weights['norm_attn'] = layer.input_layernorm.weight.detach()
        weights['norm_ffn'] = layer.post_attention_layernorm.weight.detach()
    except AttributeError:
        try:
            weights['norm_attn'] = layer.ln_1.weight.detach()
            weights['norm_ffn'] = layer.ln_2.weight.detach()
        except AttributeError:
            weights['norm_attn'] = None
            weights['norm_ffn'] = None
    
    return weights


def fold_metric_from_weights(WQ: torch.Tensor, WK: torch.Tensor, target_dim: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Rust fold_metric_svd 바인딩 사용 (SVD 기반 차원 축소)
    """
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")
    
    WQ_np = WQ.cpu().float().numpy()
    WK_np = WK.cpu().float().numpy()
    
    if WK.size(0) < WQ.size(0):
        repeat = WQ.size(0) // WK.size(0)
        WK_np = np.tile(WK_np, (repeat, 1))
    
    U, S, V, curvature = rs._rust.fold_metric_svd(WQ_np, WK_np, target_dim)
    
    return U, S, V, curvature


def build_global_metric_basis_from_model(
    model,
    target_rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 레이어의 WQ, WK를 모아 global metric basis 추출.

    구현 아이디어:
    - 레이어별 g_ell = WQ^T WK 를 계산
    - g_global = sum_ell g_ell
    - proxy WQ = I, WK = g_global 로 두고 fold_metric_svd 호출
    """
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")

    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    d_model = layers[0].self_attn.q_proj.weight.size(1)
    g_global = torch.zeros(d_model, d_model, dtype=torch.float32)

    for layer in tqdm(layers, desc="Accumulating global metric", ncols=80):
        WQ = layer.self_attn.q_proj.weight.detach().to(torch.float32)
        WK = layer.self_attn.k_proj.weight.detach().to(torch.float32)

        # GQA 처리: WK를 WQ 첫 차원에 맞게 반복
        if WK.size(0) < WQ.size(0):
            repeat = WQ.size(0) // WK.size(0)
            WK = WK.repeat(repeat, 1)

        g_layer = torch.matmul(WQ.t(), WK)
        g_global += g_layer

    # proxy WQ = I, WK = g_global 로 두고 Rust SVD 사용
    I = torch.eye(d_model, dtype=torch.float32)
    WQ_np = I.cpu().numpy()
    WK_np = g_global.cpu().numpy()

    U, S, V, _ = rs._rust.fold_metric_svd(WQ_np, WK_np, int(target_rank))
    return U, S, V


def build_global_ffn_basis_from_model(
    model,
    target_rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 레이어의 W1, W2를 모아 global FFN basis 추출.
    """
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")

    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    W1_list: List[torch.Tensor] = []
    W2_list: List[torch.Tensor] = []

    for layer in tqdm(layers, desc="Accumulating global FFN", ncols=80):
        W1 = layer.mlp.gate_proj.weight.detach().to(torch.float32)
        W2 = layer.mlp.down_proj.weight.detach().to(torch.float32)
        W1_list.append(W1)
        W2_list.append(W2)

    W1_global = torch.cat(W1_list, dim=0)
    W2_global = torch.cat(W2_list, dim=0)

    U1, S1, V1, U2, S2, V2 = rs._rust.fold_ffn(
        W1_global.cpu().numpy(),
        W2_global.cpu().numpy(),
        int(target_rank),
    )

    return U1, S1, V1, U2, S2, V2


def estimate_global_compression(
    model,
    metric_rank: int,
    ffn_rank: int,
) -> Dict[str, float]:
    """
    대략적인 global basis 기준 압축률 추정 (Step 0: 설계 검증용).

    가정:
    - 원본: 레이어당 ~4 d^2 (Q,K,V,O) + 2 d d_ff (FFN)
    - RS-ULF:
      - global metric/FFN basis: O(d r_m + d_ff r_f) 한 번
      - 레이어별 scale/곡률: O(L r_small) 는 여기서 r_small ~= r_m, r_f 로 근사
    """
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    L = len(layers)
    d_model = layers[0].self_attn.q_proj.weight.size(1)
    d_ff = layers[0].mlp.gate_proj.weight.size(0)

    # 원본 파라미터 대략치
    per_layer_attn = 4 * d_model * d_model
    per_layer_ffn = 2 * d_model * d_ff
    original_total = L * (per_layer_attn + per_layer_ffn)

    # RS-ULF 파라미터 대략치 (global basis + per-layer scale)
    r_m = metric_rank
    r_f = ffn_rank

    global_metric_params = d_model * r_m + d_model * r_m + r_m  # U, V, S
    global_ffn_params = (d_model + d_ff) * r_f * 2             # U1,V1,U2,V2 대략
    per_layer_scales = L * (r_m + r_f + 4)                     # scale + K_error + hyper 등

    rs_total = global_metric_params + global_ffn_params + per_layer_scales

    compression = original_total / max(rs_total, 1)

    return {
        "original_params_est": float(original_total),
        "rsulf_params_est": float(rs_total),
        "compression_est": float(compression),
        "d_model": int(d_model),
        "d_ff": int(d_ff),
        "num_layers": int(L),
        "metric_rank": int(r_m),
        "ffn_rank": int(r_f),
    }


def compute_global_scales_from_model(
    model,
    metric_rank: int,
    ffn_rank: int,
) -> Dict[str, object]:
    """
    Global basis + per-layer 'thin' 스케일 행렬을 계산.

    - Metric:
      - global V_metric (d, r_m)
      - per-layer D_Q^(ell) = WQ^(ell) @ V_metric (d_q, r_m)
      - per-layer D_K^(ell) = WK^(ell) @ V_metric (d_k, r_m)
    - FFN:
      - global V1 (d, r_f), V2 (d_ff, r_f)
      - per-layer D1^(ell) = W1^(ell) @ V1 (d_ff, r_f)
      - D2^(ell) = W2^(ell) @ V2 (d, r_f)
    """
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")

    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    # Global bases
    U_metric, S_metric, V_metric = build_global_metric_basis_from_model(
        model, target_rank=metric_rank
    )
    U1, S1, V1, U2, S2, V2 = build_global_ffn_basis_from_model(
        model, target_rank=ffn_rank
    )

    V_metric_torch = torch.from_numpy(V_metric).float()  # (d, r_m)
    V1_torch = torch.from_numpy(V1).float()              # (d, r_f)
    V2_torch = torch.from_numpy(V2).float()              # (d_ff, r_f)

    per_layer: List[Dict[str, np.ndarray]] = []
    k_errors_metric: List[float] = []
    k_errors_ffn: List[float] = []

    for layer in tqdm(layers, desc="Per-layer thin scales", ncols=80):
        WQ = layer.self_attn.q_proj.weight.detach().to(torch.float32)  # (d_q, d)
        WK = layer.self_attn.k_proj.weight.detach().to(torch.float32)  # (d_k, d)
        W1 = layer.mlp.gate_proj.weight.detach().to(torch.float32)     # (d_ff, d)
        W2 = layer.mlp.down_proj.weight.detach().to(torch.float32)     # (d, d_ff)

        # Metric thin factors
        D_Q = (WQ @ V_metric_torch).cpu().numpy()  # (d_q, r_m)
        D_K = (WK @ V_metric_torch).cpu().numpy()  # (d_k, r_m)

        # Reconstruction error (metric side)
        WQ_recon = torch.from_numpy(D_Q).to(WQ.dtype) @ V_metric_torch.t()
        WK_recon = torch.from_numpy(D_K).to(WK.dtype) @ V_metric_torch.t()
        err_q = (WQ - WQ_recon).norm() / (WQ.norm() + 1e-6)
        err_k = (WK - WK_recon).norm() / (WK.norm() + 1e-6)
        k_errors_metric.append(float(max(err_q.item(), err_k.item())))

        # FFN thin factors
        D1 = (W1 @ V1_torch).cpu().numpy()         # (d_ff, r_f)
        D2 = (W2 @ V2_torch).cpu().numpy()         # (d, r_f)

        W1_recon = torch.from_numpy(D1).to(W1.dtype) @ V1_torch.t()
        W2_recon = torch.from_numpy(D2).to(W2.dtype) @ V2_torch.t()
        err_w1 = (W1 - W1_recon).norm() / (W1.norm() + 1e-6)
        err_w2 = (W2 - W2_recon).norm() / (W2.norm() + 1e-6)
        k_errors_ffn.append(float(max(err_w1.item(), err_w2.item())))

        per_layer.append(
            {
                "D_Q": D_Q,
                "D_K": D_K,
                "D1": D1,
                "D2": D2,
            }
        )

    return {
        "metric_basis": {
            "U": U_metric,
            "S": S_metric,
            "V": V_metric,
        },
        "ffn_basis": {
            "U1": U1,
            "S1": S1,
            "V1": V1,
            "U2": U2,
            "S2": S2,
            "V2": V2,
        },
        "per_layer": per_layer,
        "metric_error_max": max(k_errors_metric) if k_errors_metric else 0.0,
        "ffn_error_max": max(k_errors_ffn) if k_errors_ffn else 0.0,
    }


def reconstruct_layer_weights_from_scales(
    scales: Dict[str, np.ndarray],
    metric_basis: Dict[str, np.ndarray],
    ffn_basis: Dict[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    global basis + thin scales -> approximate WQ, WK, W1, W2 재구성.
    """
    V_metric = torch.from_numpy(metric_basis["V"]).to(device=device, dtype=torch.float32)  # (d, r_m)
    V1 = torch.from_numpy(ffn_basis["V1"]).to(device=device, dtype=torch.float32)          # (d, r_f)
    V2 = torch.from_numpy(ffn_basis["V2"]).to(device=device, dtype=torch.float32)          # (d_ff, r_f)

    D_Q = torch.from_numpy(scales["D_Q"]).to(device=device, dtype=torch.float32)
    D_K = torch.from_numpy(scales["D_K"]).to(device=device, dtype=torch.float32)
    D1 = torch.from_numpy(scales["D1"]).to(device=device, dtype=torch.float32)
    D2 = torch.from_numpy(scales["D2"]).to(device=device, dtype=torch.float32)

    WQ = (D_Q @ V_metric.t()).to(dtype=dtype)
    WK = (D_K @ V_metric.t()).to(dtype=dtype)
    W1 = (D1 @ V1.t()).to(dtype=dtype)
    W2 = (D2 @ V2.t()).to(dtype=dtype)

    return WQ, WK, W1, W2


def create_causal_laplacian(seq_len: int, window: int = 8) -> np.ndarray:
    """
    Rust build_causal_laplacian 바인딩 사용
    """
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")
    
    return rs._rust.build_causal_laplacian(seq_len, window)


def create_graph_laplacian(
    seq_len: int,
    window_size: int = 8,
    directed: bool = True,
) -> torch.Tensor:
    """
    Step-by-step 스크립트 호환용 래퍼.
    내부적으로는 causal Laplacian을 생성하고 torch.Tensor로 반환.
    """
    # 현재 구현에서는 directed 플래그는 무시하고 causal 구조만 사용
    L_np = create_causal_laplacian(seq_len=seq_len, window=window_size)
    return torch.from_numpy(L_np).float()


class RSULFLayer:
    """
    RS-ULF Layer (Rust PyRSULFLayer 래퍼)
    
    논문 목표:
    - 시간복잡도: O(n²d) → O(nd)
    - 공간복잡도: O(n²) → O(d)
    - Attention 완전 제거
    """
    def __init__(
        self, 
        WQ: torch.Tensor,
        WK: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        d_model: int,
        r: int = 1024,
        eta: float = 0.01,
        alpha: float = 0.02,
        beta: float = 0.01,
        gamma: float = 0.99,
        seq_len: int = 128,
        window: int = 8,
        fast_mode: bool = False,
    ):
        if not rs._has_rust_ext:
            raise RuntimeError("Rust extension not available")
        
        WQ_np = WQ.cpu().float().numpy()
        WK_np = WK.cpu().float().numpy()
        W1_np = W1.cpu().float().numpy()
        W2_np = W2.cpu().float().numpy()
        
        if WK.size(0) < WQ.size(0):
            repeat = WQ.size(0) // WK.size(0)
            WK_np = np.tile(WK_np, (repeat, 1))
        
        if fast_mode:
            self.inner = rs._rust.PyRSULFLayer.new_fast(
                WQ_np, WK_np, W1_np, W2_np,
                d_model, r, eta, alpha, beta, gamma, seq_len, window
            )
        else:
            self.inner = rs._rust.PyRSULFLayer(
                WQ_np, WK_np, W1_np, W2_np,
                d_model, r, eta, alpha, beta, gamma, seq_len, window
            )
        self.d_model = d_model
        self.r = r
        self.device = WQ.device
        self.dtype = WQ.dtype
    
    def forward(self, x: torch.Tensor, v_mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, D) 또는 (B, L, D)
        v_mem: (B,) 또는 (B*L,) 1D 메모리 벡터 (선택적)
        """
        original_shape = x.shape
        
        # 3D 입력이면 (B*L, D)로 flatten 후 Rust 레이어에 전달
        if x.dim() == 3:
            b, l, d = x.shape
            x_flat = x.reshape(b * l, d)
        else:
            x_flat = x
        
        x_np = x_flat.cpu().float().numpy()
        
        # v_mem은 1D로 flatten 된 경우에만 전달, 아니면 None 처리
        if v_mem is not None and v_mem.dim() == 1 and v_mem.numel() == x_flat.size(0):
            v_np = v_mem.cpu().float().numpy()
        else:
            v_np = None
        
        output_np, v_new_np = self.inner.forward(x_np, v_np)
        
        output_flat = torch.from_numpy(output_np).to(self.device).to(self.dtype)
        v_new = torch.from_numpy(v_new_np).to(self.device).float()
        
        if len(original_shape) == 3:
            b, l, d = original_shape
            output = output_flat.reshape(b, l, d)
        else:
            output = output_flat
        
        return output, v_new
    
    def param_count(self) -> Dict[str, any]:
        compressed, original, ratio = self.inner.param_count()
        return {
            'compressed': compressed,
            'original': original,
            'ratio': ratio,
        }
    
    @property
    def curvature(self) -> float:
        return self.inner.curvature


class RSULFModel:
    """
    RS-ULF 모델 (전체 레이어 스택)
    """
    def __init__(self, layers: list):
        self.layers = layers
        self.d_model = layers[0].d_model if layers else 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        v = None
        for layer in self.layers:
            h, v = layer.forward(h, v)
        return h
    
    def param_count(self) -> Dict[str, any]:
        total_compressed = sum(l.param_count()['compressed'] for l in self.layers)
        total_original = sum(l.param_count()['original'] for l in self.layers)
        return {
            'compressed': total_compressed,
            'original': total_original,
            'ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'num_layers': len(self.layers),
        }

    def forward_numpy(self, x_np: np.ndarray) -> np.ndarray:
        original_shape = x_np.shape
        if x_np.ndim == 3:
            b, l, d = original_shape
            x_flat = x_np.reshape(b * l, d)
        else:
            x_flat = x_np
        h_np = x_flat
        v_np = None
        for layer in self.layers:
            h_np, v_np = layer.inner.forward(h_np, v_np)
        if len(original_shape) == 3:
            h_np = h_np.reshape(b, l, d)
        return h_np


def convert_transformer_to_rsulf(
    model,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    fast_mode: bool = False,
) -> RSULFModel:
    """
    Transformer -> RS-ULF 완전 변환
    
    Args:
        model: Huggingface Transformer model
        r: 축소 차원 (fold ratio = d_model / r)
        eta: Potential gradient 학습률
        alpha: Riemannian smoothing 계수
        beta: Graph diffusion 계수
        gamma: Bellman memory 감쇠율
        seq_len: 시퀀스 길이
        window: Laplacian window 크기
        fast_mode: True면 diagonal metric + random projection (SVD 없음, 10x 빠름)
    
    Returns:
        RSULFModel
    """
    config = model.config
    num_layers = len(model.model.layers)
    d_model = config.hidden_size
    
    mode_str = "fast" if fast_mode else "svd"
    layers = []
    for i in tqdm(range(num_layers), desc=f"RS-ULF 변환 (r={r}, {mode_str})", ncols=80):
        weights = extract_transformer_layer_weights(model, i)
        layer = RSULFLayer(
            WQ=weights['WQ'],
            WK=weights['WK'],
            W1=weights['W1'],
            W2=weights['W2'],
            d_model=d_model,
            r=r,
            eta=eta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seq_len=seq_len,
            window=window,
            fast_mode=fast_mode,
        )
        layers.append(layer)
    
    return RSULFModel(layers)


def build_rsulf_model_from_global_scales(
    model,
    scales: Dict[str, object],
    r: int = 64,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> RSULFModel:
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if dtype is None:
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    d_model = layers[0].self_attn.q_proj.weight.size(1)

    metric_basis = scales["metric_basis"]
    ffn_basis = scales["ffn_basis"]
    per_layer_scales = scales["per_layer"]

    rs_layers: List[RSULFLayer] = []

    for idx, layer_scales in enumerate(tqdm(per_layer_scales, desc="RS-ULF layers from global scales", ncols=80)):
        WQ, WK, W1, W2 = reconstruct_layer_weights_from_scales(
            layer_scales,
            metric_basis=metric_basis,
            ffn_basis=ffn_basis,
            device=device,
            dtype=dtype,
        )

        rs_layer = RSULFLayer(
            WQ=WQ,
            WK=WK,
            W1=W1,
            W2=W2,
            d_model=d_model,
            r=r,
            eta=eta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seq_len=seq_len,
            window=window,
        )
        rs_layers.append(rs_layer)

    return RSULFModel(rs_layers)


def _estimate_rsulf_layer_compression_from_shapes(
    d_model: int,
    ffn_dim: int,
    r: int,
    seq_len: int,
) -> float:
    original_attn = 4 * d_model * d_model
    original_ffn = 2 * d_model * ffn_dim + ffn_dim * d_model
    original = original_attn + original_ffn
    compressed_metric = 2 * d_model * r + r
    compressed_ffn = 2 * (ffn_dim * r + d_model * r + r)
    compressed_laplacian = seq_len * seq_len
    compressed = compressed_metric + compressed_ffn + compressed_laplacian
    return float(original) / float(compressed)


def _solve_rank_for_target_compression(
    d_model: int,
    ffn_dim: int,
    target_compression: float,
    seq_len: int,
) -> int:
    if target_compression <= 0.0:
        return d_model
    num = 4 * d_model * d_model + 3 * d_model * ffn_dim
    denom = 4 * d_model + 2 * ffn_dim + 3
    rhs = num / target_compression - float(seq_len * seq_len)
    if rhs <= 0.0:
        return 1
    r = int(rhs / float(denom))
    if r < 1:
        r = 1
    if r > d_model:
        r = d_model
    return r


def build_rsulf_model_for_target_compression(
    model,
    target_compression: float = 200.0,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
) -> Tuple[RSULFModel, Dict[str, float]]:
    weights0 = extract_transformer_layer_weights(model, 0)
    d_model = int(weights0["WQ"].size(1))
    ffn_dim = int(weights0["W1"].size(0))
    r = _solve_rank_for_target_compression(d_model, ffn_dim, target_compression, seq_len)
    theoretical_ratio = _estimate_rsulf_layer_compression_from_shapes(d_model, ffn_dim, r, seq_len)
    rs_model = convert_transformer_to_rsulf(
        model,
        r=r,
        eta=eta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seq_len=seq_len,
        window=window,
    )
    stats = rs_model.param_count()
    actual_ratio = float(stats["ratio"])
    info: Dict[str, float] = {
        "target_compression": float(target_compression),
        "theoretical_compression": float(theoretical_ratio),
        "actual_compression": float(actual_ratio),
        "d_model": float(d_model),
        "ffn_dim": float(ffn_dim),
        "rank": float(r),
        "num_layers": float(stats["num_layers"]),
        "compressed_params": float(stats["compressed"]),
        "original_params": float(stats["original"]),
    }
    if actual_ratio < target_compression:
        raise ValueError(
            f"RS-ULF compression ratio {actual_ratio:.2f}x is below target {target_compression:.2f}x"
        )
    return rs_model, info


def save_rsulf_layer_checkpoint(layer: RSULFLayer, path: str, layer_idx: int) -> None:
    import os
    comp = layer.inner.export_components()
    ckpt = {
        "d_model": comp["d_model"],
        "r": comp["r"],
        "eta": comp["eta"],
        "alpha": comp["alpha"],
        "beta": comp["beta"],
        "gamma": comp["gamma"],
        "seq_len": comp["seq_len"],
        "window": comp["window"],
        "g_diag": np.array(comp["g_diag"]),
        "g_inv": np.array(comp["g_inv"]),
        "u_metric": np.array(comp["u_metric"]),
        "v_metric": np.array(comp["v_metric"]),
        "curvature": comp["curvature"],
        "ffn_u1": np.array(comp["ffn_u1"]),
        "ffn_s1": np.array(comp["ffn_s1"]),
        "ffn_v1": np.array(comp["ffn_v1"]),
        "ffn_u2": np.array(comp["ffn_u2"]),
        "ffn_s2": np.array(comp["ffn_s2"]),
        "ffn_v2": np.array(comp["ffn_v2"]),
        "layer_idx": layer_idx,
    }
    os.makedirs(path, exist_ok=True)
    np.savez_compressed(os.path.join(path, f"layer_{layer_idx:03d}.npz"), **ckpt)


def load_rsulf_layer_checkpoint(path: str, layer_idx: int) -> RSULFLayer:
    import os
    ckpt = np.load(os.path.join(path, f"layer_{layer_idx:03d}.npz"))
    inner = rs._rust.PyRSULFLayer.from_components(
        d_model=int(ckpt["d_model"]),
        r=int(ckpt["r"]),
        eta=float(ckpt["eta"]),
        alpha=float(ckpt["alpha"]),
        beta=float(ckpt["beta"]),
        gamma=float(ckpt["gamma"]),
        seq_len=int(ckpt["seq_len"]),
        window=int(ckpt["window"]),
        g_diag=ckpt["g_diag"].astype(np.float32),
        g_inv=ckpt["g_inv"].astype(np.float32),
        u_metric=ckpt["u_metric"].astype(np.float32),
        v_metric=ckpt["v_metric"].astype(np.float32),
        curvature=float(ckpt["curvature"]),
        ffn_u1=ckpt["ffn_u1"].astype(np.float32),
        ffn_s1=ckpt["ffn_s1"].astype(np.float32),
        ffn_v1=ckpt["ffn_v1"].astype(np.float32),
        ffn_u2=ckpt["ffn_u2"].astype(np.float32),
        ffn_s2=ckpt["ffn_s2"].astype(np.float32),
        ffn_v2=ckpt["ffn_v2"].astype(np.float32),
    )
    layer = object.__new__(RSULFLayer)
    layer.inner = inner
    layer.d_model = int(ckpt["d_model"])
    layer.r = int(ckpt["r"])
    layer.device = torch.device("cpu")
    layer.dtype = torch.float32
    return layer


def save_rsulf_model_checkpoint(rs_model: RSULFModel, path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)
    for idx, layer in enumerate(rs_model.layers):
        save_rsulf_layer_checkpoint(layer, path, idx)
    meta = {
        "num_layers": len(rs_model.layers),
        "d_model": rs_model.d_model,
    }
    np.savez(os.path.join(path, "meta.npz"), **meta)


def load_rsulf_model_checkpoint(path: str) -> RSULFModel:
    import os
    meta = np.load(os.path.join(path, "meta.npz"))
    num_layers = int(meta["num_layers"])
    layers = []
    for idx in range(num_layers):
        layer = load_rsulf_layer_checkpoint(path, idx)
        layers.append(layer)
    return RSULFModel(layers)


def convert_transformer_to_rsulf_with_checkpoint(
    model,
    checkpoint_dir: str,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    resume: bool = True,
    fast_mode: bool = False,
) -> RSULFModel:
    import os
    config = model.config
    num_layers = len(model.model.layers)
    d_model = config.hidden_size
    os.makedirs(checkpoint_dir, exist_ok=True)
    existing = set()
    if resume:
        for f in os.listdir(checkpoint_dir):
            if f.startswith("layer_") and f.endswith(".npz"):
                idx_str = f.replace("layer_", "").replace(".npz", "")
                try:
                    existing.add(int(idx_str))
                except ValueError:
                    pass
    layers = [None] * num_layers
    for idx in existing:
        if idx < num_layers:
            layers[idx] = load_rsulf_layer_checkpoint(checkpoint_dir, idx)
    to_convert = [i for i in range(num_layers) if layers[i] is None]
    mode_str = "fast" if fast_mode else "svd"
    if to_convert:
        for i in tqdm(to_convert, desc=f"RS-ULF 변환 (r={r}, {mode_str})", ncols=80):
            weights = extract_transformer_layer_weights(model, i)
            layer = RSULFLayer(
                WQ=weights['WQ'],
                WK=weights['WK'],
                W1=weights['W1'],
                W2=weights['W2'],
                d_model=d_model,
                r=r,
                eta=eta,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                seq_len=seq_len,
                window=window,
                fast_mode=fast_mode,
            )
            save_rsulf_layer_checkpoint(layer, checkpoint_dir, i)
            layers[i] = layer
    meta = {"num_layers": num_layers, "d_model": d_model}
    np.savez(os.path.join(checkpoint_dir, "meta.npz"), **meta)
    return RSULFModel(layers)


def rsulf_generate(
    model,
    rs_model: RSULFModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device("cpu")

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    embed_weight = base.embed_tokens.weight.detach().to(torch_device, dtype=torch.float32)
    lm_head_weight = model.lm_head.weight.detach().to(torch_device, dtype=torch.float32)
    lm_head_bias = None
    if getattr(model.lm_head, "bias", None) is not None:
        lm_head_bias = model.lm_head.bias.detach().to(torch_device, dtype=torch.float32)

    norm_module = None
    if hasattr(base, "norm"):
        norm_module = base.norm.to(torch_device)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(torch_device)

    generated = input_ids

    def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            if mask.any():
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    pbar = tqdm(range(max_new_tokens), desc="RS-ULF generation", ncols=80)

    for _ in pbar:
        with torch.no_grad():
            embeddings = F.embedding(generated, embed_weight)
            x_np = embeddings.detach().cpu().numpy()
            h_np = rs_model.forward_numpy(x_np)
            h_torch = torch.from_numpy(h_np).to(torch_device)
            if norm_module is not None:
                h_torch = norm_module(h_torch)
            logits = F.linear(h_torch[:, -1, :], lm_head_weight, lm_head_bias)

        next_token = sample_next_token(logits[0])
        next_token = next_token.to(generated.device)
        if next_token.dim() == 0:
            next_token = next_token.unsqueeze(0)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text

