import torch
import torch.nn as nn
import reality_stone as rs
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np

try:
    from reality_stone.conversion import RiemannianLinear, RiemannianConv1D
except ImportError:
    RiemannianLinear = None
    RiemannianConv1D = None


def _is_target_layer(module):
    """리만 변환된 레이어와 원본 레이어 모두 인식"""
    if isinstance(module, nn.Linear):
        return True
    if "Conv1D" in str(type(module)):
        return True
    if RiemannianLinear and isinstance(module, RiemannianLinear):
        return True
    if RiemannianConv1D and isinstance(module, RiemannianConv1D):
        return True
    return False


def _get_weight_shape(module):
    """레이어 종류에 따라 (rows, cols) 반환"""
    if isinstance(module, nn.Linear):
        return module.weight.shape[0], module.weight.shape[1]
    if RiemannianLinear and isinstance(module, RiemannianLinear):
        return module.weight.shape[0], module.weight.shape[1]
    # Conv1D: weight shape is (in, out)
    return module.weight.shape[1], module.weight.shape[0]


def _get_weight_numpy(module):
    """레이어에서 가중치를 numpy로 추출 (rows x cols)"""
    if isinstance(module, nn.Linear):
        return module.weight.detach().float().cpu().numpy()
    if RiemannianLinear and isinstance(module, RiemannianLinear):
        return module.weight.detach().float().cpu().numpy()
    # Conv1D: transpose
    return module.weight.detach().t().float().cpu().numpy()


class RiemannianMetricExtraction(nn.Module):
    """
    리만 메트릭 추출기 (Riemannian Metric Extractor)
    
    핵심:
    Original Weight W (High-dim) ~= Basis U (Low-dim) * Metric G (Curvature) * Basis V^T
    
    G는 기하학적 튜닝을 통해 학습됨 (SVD 아님)
    """
    def __init__(self, original_layer: nn.Module, target_dim: int = 64, curvature: float = -1.0,
                 cuda_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        super().__init__()
        
        is_linear_type = isinstance(original_layer, nn.Linear) or \
                         (RiemannianLinear and isinstance(original_layer, RiemannianLinear))
        
        if is_linear_type:
            self.in_features = original_layer.weight.shape[1]
            self.out_features = original_layer.weight.shape[0]
            is_linear = True
        else:
            self.in_features = original_layer.weight.shape[0]
            self.out_features = original_layer.weight.shape[1]
            is_linear = False
            
        self.target_dim = target_dim
        self.curvature = curvature
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach().clone())
        else:
            self.bias = None
        self.is_linear = is_linear
        
        if cuda_result is not None:
            u_np, g_np, v_np = cuda_result
            device = original_layer.weight.device
            dtype = original_layer.weight.dtype
            
            self.basis_in = nn.Parameter(torch.from_numpy(v_np).to(device).to(dtype))
            self.basis_out = nn.Parameter(torch.from_numpy(u_np).to(device).to(dtype))
            self.metric_g = nn.Parameter(torch.from_numpy(g_np).to(device).to(dtype))
        else:
            self.basis_in = nn.Parameter(torch.randn(self.in_features, target_dim) * 0.02)
            self.basis_out = nn.Parameter(torch.randn(self.out_features, target_dim) * 0.02)
            self.metric_g = nn.Parameter(torch.eye(target_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.basis_in)
        z_metric = torch.matmul(z, self.metric_g)
        
        if abs(self.curvature) > 1e-5:
            z_norm = z_metric.norm(dim=-1, keepdim=True).clamp(min=1e-5)
            scale = 1.0 / (1.0 + self.curvature * z_norm)
            z_metric = z_metric * scale
            
        y = torch.matmul(z_metric, self.basis_out.t())
        
        if self.bias is not None:
            y = y + self.bias
            
        return y


def extract_riemannian_metric(
    model: nn.Module, 
    target_dim: int = 64,
    calibration_data: Optional[torch.Tensor] = None,
    num_steps: int = 100,
    curvature: float = -1.0,
    lr: float = 0.01
) -> nn.Module:
    """
    모델의 가중치 행렬에서 '리만 메트릭'을 추출합니다.
    Rust CUDA 커널로 기하학적 튜닝을 수행합니다.
    """
    print(f"Extracting Riemannian Metrics (Target Rank: {target_dim}, Steps: {num_steps})...")
    
    use_cuda = hasattr(rs, '_rust') and hasattr(rs._rust, 'extract_metric_cuda')
    if use_cuda:
        print("Using Rust CUDA backend for geometric tuning.")
    else:
        print("Warning: Rust CUDA not available. Using random initialization.")
    
    # 기본 Calibration 데이터 (없으면 랜덤)
    if calibration_data is None:
        calibration_data = torch.randn(32, 4096)
    calibration_np = calibration_data.float().cpu().numpy()
    
    count = 0
    total_orig_params = 0
    total_new_params = 0
    
    modules_to_process = []
    
    for name, module in model.named_modules():
        if not _is_target_layer(module):
            continue
        if "lm_head" in name:
            continue
        if isinstance(module, RiemannianMetricExtraction):
            continue
            
        rows, cols = _get_weight_shape(module)
        
        if min(rows, cols) > target_dim * 2:
            modules_to_process.append((name, module, rows, cols))
            
            orig_p = rows * cols
            new_p = (rows * target_dim) + (cols * target_dim) + (target_dim * target_dim)
            total_orig_params += orig_p
            total_new_params += new_p

    pbar = tqdm(modules_to_process, desc="Extracting Riemannian Metrics")
    
    for name, module, rows, cols in pbar:
        pbar.set_postfix_str(f"{name} ({rows}x{cols})")
        
        w = _get_weight_numpy(module)
        w = np.ascontiguousarray(w, dtype=np.float32)
        
        if calibration_np.shape[1] != w.shape[1]:
            calib = np.ascontiguousarray(np.random.randn(32, w.shape[1]).astype(np.float32))
        else:
            calib = np.ascontiguousarray(calibration_np, dtype=np.float32)
        
        if use_cuda:
            u, g, v = rs._rust.extract_metric_cuda(w, calib, target_dim, num_steps, curvature, lr)
        else:
            u = np.random.randn(rows, target_dim).astype(np.float32) * 0.02
            g = np.eye(target_dim, dtype=np.float32)
            v = np.random.randn(cols, target_dim).astype(np.float32) * 0.02
        
        new_layer = RiemannianMetricExtraction(module, target_dim=target_dim, curvature=curvature, cuda_result=(u, g, v))
        _replace_module(model, name, new_layer)
        count += 1
                
    if total_orig_params > 0:
        reduction = (1 - total_new_params / total_orig_params) * 100
    else:
        reduction = 0
        
    print(f"Extraction Complete. {count} layers processed.")
    print(f"   - Original Weight Params: {total_orig_params:,}")
    print(f"   - Extracted Metric Params: {total_new_params:,}")
    print(f"   - Space Saved: {reduction:.2f}%")
    
    return model

def _replace_module(root_model, path, new_module):
    atoms = path.split('.')
    parent = root_model
    try:
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_module)
    except: 
        pass

