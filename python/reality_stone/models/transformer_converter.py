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


class FoldConsistencyResult:
    def __init__(self, data: Dict[str, float]):
        self.symmetry_error = data.get("symmetry_error", 0.0)
        self.reconstruction_error = data.get("reconstruction_error", 0.0)
        self.fold_accuracy = data.get("fold_accuracy", 0.0)
        self.min_eigenvalue = data.get("min_eigenvalue", 0.0)
        self.condition_number = data.get("condition_number", float("inf"))
        self.is_valid = data.get("is_valid", False)
    
    def __repr__(self):
        return (
            f"FoldConsistency(valid={self.is_valid}, "
            f"accuracy={self.fold_accuracy:.4f}, "
            f"sym_err={self.symmetry_error:.4f}, "
            f"cond={self.condition_number:.2e})"
        )


def verify_fold_consistency(
    WQ: torch.Tensor, 
    WK: torch.Tensor, 
    target_dim: int = 128
) -> FoldConsistencyResult:
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")
    
    WQ_np = WQ.cpu().float().numpy()
    WK_np = WK.cpu().float().numpy()
    
    if WK.size(0) < WQ.size(0):
        repeat = WQ.size(0) // WK.size(0)
        WK_np = np.tile(WK_np, (repeat, 1))
    
    result = rs._rust.verify_metric_consistency(WQ_np, WK_np, target_dim)
    return FoldConsistencyResult(result)


def fold_metric_optimized(
    WQ: torch.Tensor,
    WK: torch.Tensor,
    target_dim: int = 128,
    method: str = "randomized",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, FoldConsistencyResult]:
    if not rs._has_rust_ext:
        raise RuntimeError("Rust extension not available")
    
    WQ_np = WQ.cpu().float().numpy()
    WK_np = WK.cpu().float().numpy()
    
    if WK.size(0) < WQ.size(0):
        repeat = WQ.size(0) // WK.size(0)
        WK_np = np.tile(WK_np, (repeat, 1))
    
    U, S, V, curvature, info = rs._rust.fold_metric_optimized(
        WQ_np, WK_np, target_dim, method
    )
    
    return U, S, V, curvature, FoldConsistencyResult(info)


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
    window_size: int = 8
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
    verify: bool = False,
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
        verify: 레이어별 정합성 체크 수행
    
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
        
        if verify and not fast_mode:
            consistency = verify_fold_consistency(weights['WQ'], weights['WK'], r)
            if not consistency.is_valid:
                print(f"[Layer {i}] Warning: {consistency}")
        
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
    target_compression: float = 80.0,
    fold_ratio: Optional[int] = None,
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
    if fold_ratio is not None and fold_ratio > 0:
        r = max(1, d_model // int(fold_ratio))
    else:
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
    verify_consistency: bool = True,
    min_fold_accuracy: float = 0.85,
) -> Tuple[RSULFModel, List[Dict[str, float]]]:
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
    consistency_results: List[Dict[str, float]] = []
    
    for idx in existing:
        if idx < num_layers:
            layers[idx] = load_rsulf_layer_checkpoint(checkpoint_dir, idx)
            consistency_results.append({"layer": idx, "from_checkpoint": True})
    
    to_convert = [i for i in range(num_layers) if layers[i] is None]
    mode_str = "fast" if fast_mode else "svd"
    
    if to_convert:
        for i in tqdm(to_convert, desc=f"RS-ULF 변환 (r={r}, {mode_str})", ncols=80):
            weights = extract_transformer_layer_weights(model, i)
            
            if verify_consistency and not fast_mode:
                consistency = verify_fold_consistency(
                    weights['WQ'], weights['WK'], r
                )
                result = {
                    "layer": i,
                    "symmetry_error": consistency.symmetry_error,
                    "reconstruction_error": consistency.reconstruction_error,
                    "fold_accuracy": consistency.fold_accuracy,
                    "min_eigenvalue": consistency.min_eigenvalue,
                    "condition_number": consistency.condition_number,
                    "is_valid": consistency.is_valid,
                }
                consistency_results.append(result)
                
                # 매 레이어마다 정합성 결과 출력
                status = "✓" if consistency.is_valid else "✗"
                tqdm.write(f"  [{status}] Layer {i:2d}: acc={consistency.fold_accuracy:.4f}, "
                          f"sym={consistency.symmetry_error:.4f}, cond={consistency.condition_number:.2e}")
                
                if consistency.fold_accuracy < min_fold_accuracy:
                    tqdm.write(f"      ⚠ Low fold accuracy: {consistency.fold_accuracy:.4f} < {min_fold_accuracy}")
                if not consistency.is_valid:
                    tqdm.write(f"      ⚠ Consistency check failed")
            
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
    
    return RSULFModel(layers), consistency_results


def convert_transformer_to_rsulf_with_validation(
    model,
    checkpoint_dir: str,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    svd_method: str = "randomized",
    min_fold_accuracy: float = 0.90,
    auto_adjust_rank: bool = True,
) -> Tuple[RSULFModel, Dict[str, object]]:
    import os
    config = model.config
    num_layers = len(model.model.layers)
    d_model = config.hidden_size
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    layers = []
    validation_report = {
        "layers": [],
        "failed_layers": [],
        "avg_fold_accuracy": 0.0,
        "avg_symmetry_error": 0.0,
        "total_params_compressed": 0,
        "total_params_original": 0,
    }
    
    total_accuracy = 0.0
    total_sym_error = 0.0
    
    for i in tqdm(range(num_layers), desc=f"RS-ULF 변환 (r={r}, {svd_method})", ncols=80):
        weights = extract_transformer_layer_weights(model, i)
        
        current_r = r
        best_consistency = None
        
        if auto_adjust_rank:
            for attempt_r in [r, int(r * 1.5), r * 2]:
                attempt_r = min(attempt_r, d_model)
                consistency = verify_fold_consistency(
                    weights['WQ'], weights['WK'], attempt_r
                )
                if best_consistency is None or consistency.fold_accuracy > best_consistency.fold_accuracy:
                    best_consistency = consistency
                    current_r = attempt_r
                if consistency.fold_accuracy >= min_fold_accuracy:
                    break
        else:
            best_consistency = verify_fold_consistency(
                weights['WQ'], weights['WK'], current_r
            )
        
        layer_report = {
            "layer_idx": i,
            "rank_used": current_r,
            "fold_accuracy": best_consistency.fold_accuracy,
            "symmetry_error": best_consistency.symmetry_error,
            "reconstruction_error": best_consistency.reconstruction_error,
            "condition_number": best_consistency.condition_number,
            "is_valid": best_consistency.is_valid,
        }
        validation_report["layers"].append(layer_report)
        
        total_accuracy += best_consistency.fold_accuracy
        total_sym_error += best_consistency.symmetry_error
        
        if not best_consistency.is_valid:
            validation_report["failed_layers"].append(i)
        
        layer = RSULFLayer(
            WQ=weights['WQ'],
            WK=weights['WK'],
            W1=weights['W1'],
            W2=weights['W2'],
            d_model=d_model,
            r=current_r,
            eta=eta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seq_len=seq_len,
            window=window,
            fast_mode=False,
        )
        
        save_rsulf_layer_checkpoint(layer, checkpoint_dir, i)
        layers.append(layer)
        
        stats = layer.param_count()
        validation_report["total_params_compressed"] += stats["compressed"]
        validation_report["total_params_original"] += stats["original"]
    
    validation_report["avg_fold_accuracy"] = total_accuracy / max(num_layers, 1)
    validation_report["avg_symmetry_error"] = total_sym_error / max(num_layers, 1)
    validation_report["compression_ratio"] = (
        validation_report["total_params_original"] / 
        max(validation_report["total_params_compressed"], 1)
    )
    
    meta = {"num_layers": num_layers, "d_model": d_model}
    np.savez(os.path.join(checkpoint_dir, "meta.npz"), **meta)
    
    return RSULFModel(layers), validation_report


class MetricFineTuner:
    def __init__(
        self,
        rs_layer: RSULFLayer,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
    ):
        self.rs_layer = rs_layer
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity_g = None
        self.velocity_eta = 0.0
        self.velocity_alpha = 0.0
    
    def compute_riemannian_gradient(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[np.ndarray, float, float]:
        x_np = x.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        
        if x_np.ndim == 3:
            b, l, d = x_np.shape
            x_np = x_np.reshape(b * l, d)
            target_np = target_np.reshape(b * l, d)
        
        output_np, _ = self.rs_layer.inner.forward(x_np, None)
        
        error = output_np - target_np
        
        g_diag = np.array(self.rs_layer.inner.export_components()["g_diag"])
        
        grad_g = np.zeros_like(g_diag)
        for i in range(len(g_diag)):
            grad_g[i] = np.mean(error[:, i] ** 2)
        
        grad_eta = np.mean(error ** 2)
        grad_alpha = np.mean(np.abs(error))
        
        return grad_g, grad_eta, grad_alpha
    
    def step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        grad_g, grad_eta, grad_alpha = self.compute_riemannian_gradient(x, target)
        
        if self.velocity_g is None:
            self.velocity_g = np.zeros_like(grad_g)
        
        self.velocity_g = self.momentum * self.velocity_g + grad_g
        self.velocity_eta = self.momentum * self.velocity_eta + grad_eta
        self.velocity_alpha = self.momentum * self.velocity_alpha + grad_alpha
        
        comp = self.rs_layer.inner.export_components()
        g_diag = np.array(comp["g_diag"])
        eta = float(comp["eta"])
        alpha = float(comp["alpha"])
        
        g_diag_new = g_diag - self.lr * self.velocity_g
        g_diag_new = np.clip(g_diag_new, 1e-6, 1e6)
        
        eta_new = eta - self.lr * 0.01 * self.velocity_eta
        eta_new = np.clip(eta_new, 0.001, 0.05)
        
        alpha_new = alpha - self.lr * 0.01 * self.velocity_alpha
        alpha_new = np.clip(alpha_new, 0.001, 0.05)
        
        return {
            "grad_g_norm": float(np.linalg.norm(grad_g)),
            "grad_eta": float(grad_eta),
            "grad_alpha": float(grad_alpha),
            "eta": float(eta_new),
            "alpha": float(alpha_new),
        }


def finetune_rsulf_layer(
    rs_layer: RSULFLayer,
    data_loader,
    num_steps: int = 100,
    learning_rate: float = 1e-4,
) -> Dict[str, List[float]]:
    tuner = MetricFineTuner(rs_layer, learning_rate=learning_rate)
    history = {
        "grad_g_norm": [],
        "grad_eta": [],
        "grad_alpha": [],
    }
    
    step = 0
    for batch in data_loader:
        if step >= num_steps:
            break
        
        x = batch["input"] if isinstance(batch, dict) else batch[0]
        target = batch["target"] if isinstance(batch, dict) else batch[1]
        
        metrics = tuner.step(x, target)
        
        for key in history:
            if key in metrics:
                history[key].append(metrics[key])
        
        step += 1
    
    return history


def optimize_metric_extraction(
    WQ: torch.Tensor,
    WK: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    target_dim: int,
    num_calibration_steps: int = 50,
    learning_rate: float = 0.1,
    num_samples: int = 16,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    리만 메트릭 추출 + 최적화 보정
    
    Returns:
        g_diag: 최적화된 대각 메트릭
        g_inv: 역메트릭
        final_loss: 최종 손실값
    """
    d_model = WQ.size(1)
    
    # GPU로 이동
    WQ = WQ.to(device).float()
    WK = WK.to(device).float()
    W1 = W1.to(device).float()
    W2 = W2.to(device).float()
    
    # GQA 처리
    if WK.size(0) < WQ.size(0):
        repeat = WQ.size(0) // WK.size(0)
        WK = WK.repeat(repeat, 1)
    
    # 1. 초기 대각 메트릭: g_ii = WQ_i · WK_i (부호 유지)
    with torch.no_grad():
        g_diag_init = (WQ * WK).sum(dim=0)  # (d_model,)
        g_diag_init = torch.clamp(g_diag_init.abs(), min=1e-6, max=1e6)
        
        # 정규화: 평균 1로
        g_mean = g_diag_init.mean()
        g_diag_init = g_diag_init / (g_mean + 1e-8)
    
    # 2. 학습 가능한 스케일 (작은 범위)
    log_scale = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.SGD([log_scale], lr=learning_rate, momentum=0.9)
    
    # 3. Calibration 루프
    best_loss = float('inf')
    best_scale = None
    
    for step in range(num_calibration_steps):
        optimizer.zero_grad()
        
        # 현재 메트릭 (스케일 범위 제한: 0.1 ~ 10)
        scale = torch.sigmoid(log_scale) * 9.9 + 0.1
        g_diag = g_diag_init * scale
        
        # 랜덤 샘플
        x = torch.randn(num_samples, d_model, device=device, dtype=torch.float32)
        y = torch.randn(num_samples, d_model, device=device, dtype=torch.float32)
        
        # Inner-product 보존 손실 (정규화)
        with torch.no_grad():
            Qx = F.linear(x, WQ)
            Ky = F.linear(y, WK)
            ip_target = (Qx * Ky).sum(dim=-1)
            ip_target_norm = ip_target / (ip_target.abs().mean() + 1e-8)
        
        ip_pred = (x * (g_diag.unsqueeze(0) * y)).sum(dim=-1)
        ip_pred_norm = ip_pred / (ip_pred.abs().mean() + 1e-8)
        
        # Cosine similarity 기반 손실 (스케일 불변)
        loss_ip = 1.0 - F.cosine_similarity(ip_pred_norm.unsqueeze(0), ip_target_norm.unsqueeze(0)).mean()
        
        # 정규화 손실
        loss_reg = 0.001 * ((scale - 1.0) ** 2).mean()
        
        total_loss = loss_ip + loss_reg
        
        total_loss.backward()
        optimizer.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_scale = scale.detach().clone()
    
    # 4. 최종 메트릭
    with torch.no_grad():
        if best_scale is None:
            best_scale = torch.ones(d_model, device=device)
        g_diag_final = (g_diag_init * best_scale * g_mean).cpu().numpy()
        g_diag_final = np.clip(g_diag_final, 1e-6, 1e6)
        g_inv_final = 1.0 / g_diag_final
    
    return g_diag_final, g_inv_final, best_loss


class RSULFLayerOptimized:
    """
    최적화된 RS-ULF Layer (메트릭 추출 시 calibration 포함)
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
        calibration_steps: int = 50,
        device: str = "cuda",
    ):
        if not rs._has_rust_ext:
            raise RuntimeError("Rust extension not available")
        
        # 1. 메트릭 최적화
        g_diag, g_inv, calib_loss = optimize_metric_extraction(
            WQ,
            WK,
            W1,
            W2,
            target_dim=r,
            num_calibration_steps=calibration_steps,
            device=device,
        )
        
        self.calibration_loss = calib_loss
        
        # 2. Rust 레이어 생성 (최적화된 메트릭 사용)
        WQ_np = WQ.cpu().float().numpy()
        WK_np = WK.cpu().float().numpy()
        W1_np = W1.cpu().float().numpy()
        W2_np = W2.cpu().float().numpy()
        
        if WK.size(0) < WQ.size(0):
            repeat = WQ.size(0) // WK.size(0)
            WK_np = np.tile(WK_np, (repeat, 1))

        # 최적화된 대각 메트릭을 Rust 레이어에 전달
        # 1순위: 새 바인딩(new_with_metric)이 있을 때 직접 사용
        # 2순위: 구버전 바인딩일 경우 export_components / from_components 경로로 주입
        try:
            self.inner = rs._rust.PyRSULFLayer.new_with_metric(
                WQ_np,
                WK_np,
                W1_np,
                W2_np,
                g_diag.astype(np.float32),
                d_model,
                r,
                eta,
                alpha,
                beta,
                gamma,
                seq_len,
                window,
            )
        except AttributeError:
            # 구버전: 먼저 기본 레이어를 만들고, export_components 후 from_components로 교체
            base = rs._rust.PyRSULFLayer(
                WQ_np,
                WK_np,
                W1_np,
                W2_np,
                d_model,
                r,
                eta,
                alpha,
                beta,
                gamma,
                seq_len,
                window,
            )
            comp = base.export_components()

            self.inner = rs._rust.PyRSULFLayer.from_components(
                int(comp["d_model"]),
                int(comp["r"]),
                float(comp["eta"]),
                float(comp["alpha"]),
                float(comp["beta"]),
                float(comp["gamma"]),
                int(comp["seq_len"]),
                int(comp["window"]),
                g_diag.astype(np.float32),
                g_inv.astype(np.float32),
                np.array(comp["u_metric"], dtype=np.float32),
                np.array(comp["v_metric"], dtype=np.float32),
                float(comp["curvature"]),
                np.array(comp["ffn_u1"], dtype=np.float32),
                np.array(comp["ffn_s1"], dtype=np.float32),
                np.array(comp["ffn_v1"], dtype=np.float32),
                np.array(comp["ffn_u2"], dtype=np.float32),
                np.array(comp["ffn_s2"], dtype=np.float32),
                np.array(comp["ffn_v2"], dtype=np.float32),
            )
        
        self.d_model = d_model
        self.r = r
        self.device = WQ.device
        self.dtype = WQ.dtype
        self.g_diag_optimized = g_diag
        self.g_inv_optimized = g_inv
    
    def forward(self, x: torch.Tensor, v_mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if x.dim() == 3:
            b, l, d = x.shape
            x_flat = x.reshape(b * l, d)
        else:
            x_flat = x
        
        x_np = x_flat.cpu().float().numpy()
        v_np = None
        if v_mem is not None and v_mem.dim() == 1 and v_mem.numel() == x_flat.size(0):
            v_np = v_mem.cpu().float().numpy()
        
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


def convert_transformer_to_rsulf_optimized(
    model,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    calibration_steps: int = 50,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[RSULFModel, Dict[str, float]]:
    """
    Transformer → RS-ULF 변환 (메트릭 최적화 포함)
    
    단순 SVD가 아니라 calibration을 통해 메트릭을 최적화합니다.
    """
    import os
    
    config = model.config
    num_layers = len(model.model.layers)
    d_model = config.hidden_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    layers = []
    calibration_losses = []
    
    for i in tqdm(range(num_layers), desc=f"RS-ULF 최적화 변환 (r={r}, calib={calibration_steps})", ncols=80):
        weights = extract_transformer_layer_weights(model, i)
        
        # 정합성 체크 (최적화 전)
        consistency_before = verify_fold_consistency(weights['WQ'], weights['WK'], r)
        
        # 최적화된 레이어 생성
        layer = RSULFLayerOptimized(
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
            calibration_steps=calibration_steps,
            device=device,
        )
        
        calibration_losses.append(layer.calibration_loss)
        
        tqdm.write(f"  [✓] Layer {i:2d}: calib_loss={layer.calibration_loss:.6f}, "
                  f"fold_acc={consistency_before.fold_accuracy:.4f}")
        
        if checkpoint_dir:
            # 체크포인트 저장 (RSULFLayer 호환)
            layer_compat = RSULFLayer(
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
                fast_mode=False,
            )
            save_rsulf_layer_checkpoint(layer_compat, checkpoint_dir, i)
        
        layers.append(layer)
    
    if checkpoint_dir:
        meta = {"num_layers": num_layers, "d_model": d_model}
        np.savez(os.path.join(checkpoint_dir, "meta.npz"), **meta)
    
    report = {
        "num_layers": num_layers,
        "avg_calibration_loss": sum(calibration_losses) / len(calibration_losses),
        "calibration_steps": calibration_steps,
        "r": r,
    }
    
    # RSULFModel 호환 래퍼
    rs_model = RSULFModel([])
    rs_model.layers = layers
    rs_model.d_model = d_model
    
    return rs_model, report


class LowRankFFNStudent(torch.nn.Module):
    """
    Low-rank FFN student: approximates Transformer FFN with rank-r factors.
    W1 ≈ U1 diag(S1) V1^T, W2 ≈ U2 diag(S2) V2^T.
    """

    def __init__(self, d_model: int, d_ff: int, r: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.r = r

        self.V1 = torch.nn.Parameter(torch.randn(d_model, r) / (d_model ** 0.5))
        self.S1 = torch.nn.Parameter(torch.ones(r))
        self.U1 = torch.nn.Parameter(torch.randn(d_ff, r) / (d_ff ** 0.5))

        self.V2 = torch.nn.Parameter(torch.randn(d_ff, r) / (d_ff ** 0.5))
        self.S2 = torch.nn.Parameter(torch.ones(r))
        self.U2 = torch.nn.Parameter(torch.randn(d_model, r) / (d_model ** 0.5))

    @torch.no_grad()
    def init_from_weights(self, W1: torch.Tensor, W2: torch.Tensor):
        """
        Initialize from full FFN weights using truncated SVD.
        W1: (d_ff, d_model), W2: (d_model, d_ff)
        """
        # W1 SVD
        try:
            U1_full, S1_full, V1h_full = torch.linalg.svd(W1, full_matrices=False)
        except RuntimeError:
            U1_full, S1_full, V1h_full = torch.svd(W1)
        r = min(self.r, S1_full.shape[0])
        self.U1.copy_(U1_full[:, :r])
        self.S1.data.zero_()
        self.S1[:r].copy_(S1_full[:r])
        self.V1.copy_(V1h_full[:r, :].t())

        # W2 SVD
        try:
            U2_full, S2_full, V2h_full = torch.linalg.svd(W2, full_matrices=False)
        except RuntimeError:
            U2_full, S2_full, V2h_full = torch.svd(W2)
        r2 = min(self.r, S2_full.shape[0])
        self.U2.copy_(U2_full[:, :r2])
        self.S2.data.zero_()
        self.S2[:r2].copy_(S2_full[:r2])
        self.V2.copy_(V2h_full[:r2, :].t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) or (B, L, D)
        Returns: residual output x + f_lowrank(x)
        """
        original_shape = x.shape
        if x.dim() == 3:
            b, l, d = original_shape
            x_flat = x.view(b * l, d)
        else:
            x_flat = x

        # Approximate W1: (d_ff, d_model)
        h1 = x_flat @ self.V1          # (B, r)
        h1 = h1 * self.S1              # scale by singular values
        pre_act = h1 @ self.U1.t()     # (B, d_ff)
        h_act = F.silu(pre_act)

        # Approximate W2: (d_model, d_ff)
        h2 = h_act @ self.V2           # (B, r)
        h2 = h2 * self.S2
        f_x = h2 @ self.U2.t()         # (B, d_model)

        y = x_flat + f_x
        if x.dim() == 3:
            return y.view(b, l, d)
        return y


def distill_ffn_low_rank_for_layer(
    model,
    layer_idx: int,
    r: int,
    num_steps: int = 200,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Distill a single Transformer's FFN (gate_proj + down_proj) to a low-rank student.

    Teacher: y_T = x + W2 σ(W1 x)
    Student: y_S = LowRankFFNStudent(x)
    Objective: E[||y_T - y_S||^2]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate layer (reuse logic from extract_transformer_layer_weights)
    try:
        layer = model.model.layers[layer_idx]
    except AttributeError:
        try:
            layer = model.transformer.h[layer_idx]
        except AttributeError:
            layer = model.layers[layer_idx]

    W1 = layer.mlp.gate_proj.weight.detach().to(device=device, dtype=torch.float32)
    W2 = layer.mlp.down_proj.weight.detach().to(device=device, dtype=torch.float32)

    d_ff, d_model = W1.shape
    assert W2.shape[1] == d_ff and W2.shape[0] == d_model

    student = LowRankFFNStudent(d_model=d_model, d_ff=d_ff, r=r).to(device)
    student.init_from_weights(W1, W2)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    losses: List[float] = []
    cosines: List[float] = []

    for step in range(num_steps):
        x = torch.randn(batch_size, d_model, device=device, dtype=torch.float32)

        # Teacher FFN residual
        with torch.no_grad():
            h1 = F.linear(x, W1)
            h_act = F.silu(h1)
            f_teacher = F.linear(h_act, W2)
            y_teacher = x + f_teacher

        y_student = student(x)

        loss = F.mse_loss(y_student, y_teacher)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cos = F.cosine_similarity(
                y_teacher.view(batch_size, -1),
                y_student.view(batch_size, -1),
                dim=-1,
            ).mean().item()

        losses.append(float(loss.item()))
        cosines.append(cos)

    # Reconstruct full low-rank W1_hat, W2_hat
    with torch.no_grad():
        V1 = student.V1           # (d_model, r)
        S1 = student.S1           # (r,)
        U1 = student.U1           # (d_ff, r)
        V2 = student.V2           # (d_ff, r)
        S2 = student.S2           # (r,)
        U2 = student.U2           # (d_model, r)

        W1_low = (U1 * S1.view(1, -1)) @ V1.t()  # (d_ff, d_model)
        W2_low = (U2 * S2.view(1, -1)) @ V2.t()  # (d_model, d_ff)

    info = {
        "final_loss": float(losses[-1] if losses else 0.0),
        "final_cosine": float(cosines[-1] if cosines else 0.0),
        "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "avg_cosine": float(sum(cosines) / len(cosines)) if cosines else 0.0,
        "num_steps": int(num_steps),
        "rank": int(r),
    }

    return W1_low.cpu(), W2_low.cpu(), info


def convert_transformer_to_rsulf_with_ffn_distillation(
    model,
    r: int = 1024,
    eta: float = 0.01,
    alpha: float = 0.02,
    beta: float = 0.01,
    gamma: float = 0.99,
    seq_len: int = 128,
    window: int = 8,
    distill_steps: int = 200,
    distill_batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[RSULFModel, Dict[str, object]]:
    """
    Transformer → RS-ULF 변환 (FFN 저랭크 증류 포함).

    - Q, K는 그대로 사용
    - FFN(W1, W2)는 레이어별 distillation으로 rank-r 저랭크 근사로 교체
    - 그 이후 RSULFLayer로 매핑
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        layers_tf = model.model.layers
    except AttributeError:
        layers_tf = model.layers

    num_layers = len(layers_tf)
    d_model = layers_tf[0].self_attn.q_proj.weight.size(1)

    rs_layers: List[RSULFLayer] = []
    distill_reports: List[Dict[str, object]] = []

    for idx in tqdm(range(num_layers), desc=f"RS-ULF FFN distill (r={r})", ncols=80):
        weights = extract_transformer_layer_weights(model, idx)

        # Distill FFN of this layer to rank-r student
        W1_low, W2_low, info = distill_ffn_low_rank_for_layer(
            model,
            layer_idx=idx,
            r=r,
            num_steps=distill_steps,
            batch_size=distill_batch_size,
            device=device,
        )

        # Convert distilled numpy weights back to torch on correct device/dtype
        W1_t = torch.from_numpy(W1_low).to(
            weights["W1"].device, dtype=weights["W1"].dtype
        )
        W2_t = torch.from_numpy(W2_low).to(
            weights["W2"].device, dtype=weights["W2"].dtype
        )

        # Build RSULF layer using original Q/K and distilled FFN
        rs_layer = RSULFLayer(
            WQ=weights["WQ"],
            WK=weights["WK"],
            W1=W1_t,
            W2=W2_t,
            d_model=d_model,
            r=r,
            eta=eta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seq_len=seq_len,
            window=window,
            fast_mode=False,
        )

        rs_layers.append(rs_layer)
        layer_report: Dict[str, object] = {"layer_idx": idx}
        layer_report.update(info)
        distill_reports.append(layer_report)

        tqdm.write(
            f"  [FFN distill] layer {idx:2d}: "
            f"cos={info['final_cosine']:.4f}, loss={info['final_loss']:.6f}"
        )

    rs_model = RSULFModel(rs_layers)
    stats = rs_model.param_count()

    avg_cos = (
        sum(r["final_cosine"] for r in distill_reports) / len(distill_reports)
        if distill_reports
        else 0.0
    )

    summary: Dict[str, object] = {
        "num_layers": num_layers,
        "rank": r,
        "avg_final_cosine": float(avg_cos),
        "compressed_params": float(stats["compressed"]),
        "original_params": float(stats["original"]),
        "compression_ratio": float(stats["ratio"]),
        "per_layer": distill_reports,
    }

    return rs_model, summary


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


def finetune_rsulf_lm_head(
    model,
    rs_model: RSULFModel,
    tokenizer,
    train_loader,
    num_steps: int = 1000,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    RS-ULF LLM용 lm_head만 미세조정하는 distillation 루프.

    - Transformer 본체와 RS-ULF 레이어는 고정
    - RS-ULF가 만든 hidden state 위에서 lm_head를 LM loss로 학습
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    # lm_head만 학습, 나머지는 freeze
    # RS-ULF 경로는 float32로 동작하므로 lm_head도 float32로 맞춘다.
    lm_head = model.lm_head.to(torch_device, dtype=torch.float32)
    base.to(torch_device)
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(lm_head.parameters(), lr=lr)
    loss_fct = torch.nn.CrossEntropyLoss()

    step = 0
    for batch in tqdm(train_loader, desc="Finetune RS-ULF lm_head", ncols=80):
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(torch_device)
        labels = batch.get("labels", batch["input_ids"]).to(torch_device)

        with torch.no_grad():
            # RS-ULF Rust 바인딩은 float32를 기대하므로 임베딩도 float32로 맞춘다.
            embed_weight = base.embed_tokens.weight.to(torch_device, dtype=torch.float32)
            embeddings = F.embedding(input_ids, embed_weight).to(torch.float32)  # (B, L, D)
            x_np = embeddings.detach().cpu().numpy()
            h_np = rs_model.forward_numpy(x_np)
            h_torch = torch.from_numpy(h_np).to(torch_device)
            if hasattr(base, "norm"):
                h_torch = base.norm(h_torch)

        # Shift for LM loss
        shift_hidden = h_torch[:, :-1, :]
        shift_labels = labels[:, 1:]

        logits = F.linear(shift_hidden, lm_head.weight, lm_head.bias)

        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm_head.parameters(), 1.0)
        optimizer.step()

        step += 1

    return lm_head


def cache_rsulf_hidden_states(
    model,
    rs_model,
    train_loader,
    cache_path: str,
    device: str = "cuda",
    max_samples: int = 10000,
):
    """
    RS-ULF hidden states를 미리 계산하여 디스크에 저장 (1회 실행).
    이후 lm_head 학습 시 캐시를 불러와 사용하면 훨씬 빠름.
    """
    import os
    torch_device = torch.device(device)

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model
    base.to(torch_device)

    all_hidden = []
    all_labels = []
    count = 0

    for batch in tqdm(train_loader, desc="Caching RS-ULF hidden", ncols=80):
        if count >= max_samples:
            break

        input_ids = batch["input_ids"].to(torch_device)
        labels = batch.get("labels", batch["input_ids"]).to(torch_device)

        with torch.no_grad():
            embed_weight = base.embed_tokens.weight.to(torch_device, dtype=torch.float32)
            embeddings = F.embedding(input_ids, embed_weight).to(torch.float32)
            x_np = embeddings.detach().cpu().numpy()
            h_np = rs_model.forward_numpy(x_np)
            h_torch = torch.from_numpy(h_np)
            if hasattr(base, "norm"):
                h_torch = base.norm(h_torch.to(torch_device)).cpu()

        all_hidden.append(h_torch)
        all_labels.append(labels.cpu())
        count += input_ids.size(0)

    hidden_cat = torch.cat(all_hidden, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    torch.save({"hidden": hidden_cat, "labels": labels_cat}, cache_path)
    print(f"  캐시 저장: {cache_path} ({hidden_cat.shape[0]} 샘플)")
    return cache_path


def finetune_lm_head_from_cache(
    model,
    cache_path: str,
    num_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    캐싱된 hidden states로 lm_head만 빠르게 학습.
    RS-ULF forward 없이 순수 GPU 연산만 수행.
    """
    torch_device = torch.device(device)

    cache = torch.load(cache_path, map_location="cpu")
    hidden = cache["hidden"]
    labels = cache["labels"]

    lm_head = model.lm_head.to(torch_device, dtype=torch.float32)
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(lm_head.parameters(), lr=lr)
    loss_fct = torch.nn.CrossEntropyLoss()

    n_samples = hidden.size(0)
    step = 0
    pbar = tqdm(total=num_steps, desc="Finetune lm_head (cached)", ncols=80)

    while step < num_steps:
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            if step >= num_steps:
                break
            idx = perm[i : i + batch_size]
            h_batch = hidden[idx].to(torch_device, dtype=torch.float32)
            l_batch = labels[idx].to(torch_device)

            shift_hidden = h_batch[:, :-1, :]
            shift_labels = l_batch[:, 1:]

            logits = F.linear(shift_hidden, lm_head.weight, lm_head.bias)
            loss = loss_fct(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_head.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)
            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    pbar.close()
    return lm_head
