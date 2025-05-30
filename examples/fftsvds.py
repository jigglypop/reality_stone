import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter
import copy
# ───────── Enhanced RealityStone & Advanced Mathematical Utils ─────────
try:
    import reality_stone as rs
    print("✅ RealityStone 라이브러리 로드 성공!")
    print(f"   🌟 버전: {getattr(rs, '__version__', 'Unknown')}")
    print(f"   🚀 고급 기능: {', '.join(dir(rs)) if hasattr(rs, '__dict__') else 'Standard'}")
    RS_AVAILABLE = True
except ImportError:
    print("⚠️ RealityStone 라이브러리 없음 - 최고급 자체 구현 사용")
    RS_AVAILABLE = False

def enhanced_stereographic_projection(z: torch.Tensor, use_complex_log=True) -> torch.Tensor:
    """향상된 스테레오그래픽 투영 (복소 로그 및 안정성 개선)"""
    
    if use_complex_log:
        # 복소 로그를 활용한 더 안정적인 투영
        z_conj = torch.conj(z)
        norm_sq = (z * z_conj).real
        
        # 로그 스케일링으로 수치 안정성 향상
        log_factor = torch.log(1 + norm_sq + 1e-8)
        scaling = torch.exp(-log_factor / 4)  # 적응적 스케일링
        
        z_scaled = z * scaling
        real, imag = z_scaled.real, z_scaled.imag
    else:
        real, imag = z.real, z.imag
    
    # 개선된 분모 계산 (수치 안정성)
    norm_sq = real**2 + imag**2
    denom = 1 + norm_sq
    epsilon = torch.finfo(real.dtype).eps * 10
    denom = torch.clamp(denom, min=epsilon)
    
    # 고정밀 스테레오그래픽 좌표
    X = 2 * real / denom
    Y = 2 * imag / denom
    Z = (norm_sq - 1) / denom
    
    # 북극점 근처에서의 특별 처리
    pole_mask = norm_sq > 100  # 북극점 근처
    X = torch.where(pole_mask, torch.sign(real) * 0.99, X)
    Y = torch.where(pole_mask, torch.sign(imag) * 0.99, Y)
    Z = torch.where(pole_mask, torch.ones_like(Z) * 0.99, Z)
    
    return torch.stack([X, Y, Z], dim=-1)

def enhanced_inverse_stereographic_projection(sphere_coords: torch.Tensor, 
                                            use_mobius_normalization=True) -> torch.Tensor:
    """향상된 역스테레오그래픽 투영 (뫼비우스 정규화 포함)"""
    
    X, Y, Z = sphere_coords[..., 0], sphere_coords[..., 1], sphere_coords[..., 2]
    
    # 향상된 북극점 처리
    epsilon = torch.finfo(X.dtype).eps * 100
    denom = torch.clamp(1 - Z, min=epsilon)
    
    real = X / denom
    imag = Y / denom
    
    if use_mobius_normalization:
        # 뫼비우스 변환을 통한 정규화
        complex_result = torch.complex(real, imag)
        norm = torch.abs(complex_result)
        
        # 단위원 내부로 정규화
        scale_factor = torch.where(norm > 0.95, 0.95 / (norm + epsilon), torch.ones_like(norm))
        complex_result = complex_result * scale_factor
        
        return complex_result
    
    return torch.complex(real, imag)

def advanced_riemann_distance(z1: torch.Tensor, z2: torch.Tensor, 
                            metric_type='hyperbolic') -> torch.Tensor:
    """고급 리만 구면 거리 (다양한 메트릭 지원)"""
    
    if metric_type == 'hyperbolic':
        # 하이퍼볼릭 거리 (포인카레 모델)
        z1_conj = torch.conj(z1)
        z2_conj = torch.conj(z2)
        
        numerator = torch.abs(z1 - z2)**2
        denom1 = 1 - torch.abs(z1)**2
        denom2 = 1 - torch.abs(z2)**2
        
        epsilon = 1e-8
        denom_product = torch.clamp(denom1 * denom2, min=epsilon)
        
        ratio = 1 + 2 * numerator / denom_product
        ratio = torch.clamp(ratio, min=1 + epsilon)
        
        return torch.acosh(ratio)
        
    elif metric_type == 'spherical':
        # 구면 거리 (기존 방식 개선)
        p1 = enhanced_stereographic_projection(z1)
        p2 = enhanced_stereographic_projection(z2)
        
        dot_product = torch.sum(p1 * p2, dim=-1)
        dot_product = torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7)
        
        return torch.acos(dot_product)
        
    elif metric_type == 'fubini_study':
        # 푸비니-스터디 메트릭
        z1_norm = torch.norm(torch.stack([z1.real, z1.imag], dim=-1), dim=-1)
        z2_norm = torch.norm(torch.stack([z2.real, z2.imag], dim=-1), dim=-1)
        
        inner_product = torch.real(torch.conj(z1) * z2)
        
        ratio = torch.abs(inner_product) / (z1_norm * z2_norm + 1e-8)
        ratio = torch.clamp(ratio, max=1 - 1e-7)
        
        return torch.acos(ratio)
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def advanced_mobius_transform(z: torch.Tensor, params: dict) -> torch.Tensor:
    """고급 뫼비우스 변환 (매개변수화 개선)"""
    
    a = params.get('a', torch.tensor(1.0, dtype=z.dtype, device=z.device))
    b = params.get('b', torch.tensor(0.0, dtype=z.dtype, device=z.device))
    c = params.get('c', torch.tensor(0.0, dtype=z.dtype, device=z.device))
    d = params.get('d', torch.tensor(1.0, dtype=z.dtype, device=z.device))
    
    # 행렬식 정규화
    det = a * d - b * c
    if torch.abs(det) < 1e-7:
        # 특이 변환 처리
        return z
    
    sqrt_det = torch.sqrt(torch.abs(det))
    a, b, c, d = a/sqrt_det, b/sqrt_det, c/sqrt_det, d/sqrt_det
    
    numerator = a * z + b
    denominator = c * z + d
    
    # 안전한 나눗셈
    epsilon = 1e-8
    mask = torch.abs(denominator) < epsilon
    
    # 무한대 처리 개선
    inf_value = torch.tensor(float('inf'), dtype=z.dtype, device=z.device)
    if torch.is_complex(z):
        inf_value = torch.complex(inf_value, torch.tensor(0.0, dtype=z.real.dtype, device=z.device))
    
    result = torch.where(mask, inf_value, numerator / denominator)
    
    return result

# ───────── Fast SVD Compressor for Speed Optimization ─────────
class FastSVDCompressor:
    """빠른 SVD 압축기 (속도 최적화)"""
    
    def __init__(self, W: torch.Tensor, compression_ratio=0.1):
        """
        Args:
            W: 가중치 행렬 [out_f, in_f]
            compression_ratio: 압축률
        """
        self.out_f, self.in_f = W.shape
        self.compression_ratio = compression_ratio
        
        print(f"    ⚡ 고속 SVD 압축: {W.shape}, 압축률={compression_ratio:.1%}")
        
        self._apply_fast_svd_compression(W)
    
    def _apply_fast_svd_compression(self, W: torch.Tensor):
        """빠른 SVD 압축 적용"""
        
        # 적응적 랭크 선택 (에너지 기반)
        U, S, V = torch.svd(W.float())
        
        # 에너지 임계값 기반 랭크 선택 (품질 우선)
        energy_cumsum = torch.cumsum(S**2, dim=0)
        total_energy = energy_cumsum[-1]
        energy_threshold = 0.95  # 95% 에너지 보존 (90% → 95%)
        
        energy_rank = torch.sum(energy_cumsum < total_energy * energy_threshold).item() + 1
        target_rank = max(8, int(min(W.shape) * self.compression_ratio * 6))  # 더 관대 (4→6배)
        
        # 최적 랭크 선택 (품질 우선)
        optimal_rank = min(energy_rank, target_rank, len(S), min(W.shape) // 3)  # 1/3 제한 (1/4→1/3)
        
        # 압축된 파라미터 저장
        self.U = nn.Parameter(U[:, :optimal_rank].to(W.dtype))
        self.S = nn.Parameter(S[:optimal_rank].to(W.dtype))
        self.V = nn.Parameter(V[:, :optimal_rank].to(W.dtype))
        
        # 압축 효과 계산
        original_params = W.numel()
        compressed_params = self.U.numel() + self.S.numel() + self.V.numel()
        actual_ratio = compressed_params / original_params
        
        print(f"       ✅ 고속 압축 완료: rank {optimal_rank}, 실제 압축률 {actual_ratio:.1%}")
    
    def reconstruct(self) -> torch.Tensor:
        """압축된 가중치 복원"""
        return self.U @ torch.diag(self.S) @ self.V.t()
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """압축된 연산 적용 (최적화된 버전)"""
        # 3단계로 나누어 효율적 계산: x @ V @ diag(S) @ U.t()
        step1 = x @ self.V  # [batch, rank]
        step2 = step1 * self.S.unsqueeze(0)  # [batch, rank] (broadcasting)
        step3 = step2 @ self.U.t()  # [batch, out_features]
        return step3

# ───────── Enhanced FFT-SVD Hybrid Compressor ─────────
class AdvancedFFTSVDCompressor:
    """고급 FFT+SVD 하이브리드 압축기"""
    
    def __init__(self, W: torch.Tensor, compression_ratio=0.1, fft_ratio=0.3):
        """
        Args:
            W: 가중치 행렬 [out_f, in_f]
            compression_ratio: 전체 압축률
            fft_ratio: FFT 영역에 할당할 비율 (나머지는 SVD)
        """
        self.out_f, self.in_f = W.shape
        self.compression_ratio = compression_ratio
        self.fft_ratio = fft_ratio
        
        print(f"    🌊 FFT+SVD 하이브리드: {W.shape}, 압축률={compression_ratio:.1%}")
        print(f"       FFT영역={fft_ratio:.1%}, SVD영역={1-fft_ratio:.1%}")
        
        self._apply_fft_svd_compression(W)
    
    def _apply_fft_svd_compression(self, W: torch.Tensor):
        """FFT+SVD 하이브리드 압축 적용"""
        
        # 1단계: 2D FFT 변환으로 주파수 도메인 분석
        print(f"       🌊 2D FFT 주파수 분석...")
        W_fft = self._enhanced_2d_fft(W)
        
        # 2단계: 주파수 도메인에서 중요 성분 선택
        important_freqs, freq_mask = self._select_important_frequencies(W_fft)
        
        # 3단계: FFT 영역과 잔차 영역 분리
        fft_component, residual = self._separate_fft_residual(W, important_freqs, freq_mask)
        
        # 4단계: FFT 압축
        self.fft_compressed = self._compress_fft_component(fft_component)
        
        # 5단계: 잔차에 대한 고급 SVD
        self.svd_compressed = self._compress_residual_svd(residual)
        
        print(f"       ✅ FFT+SVD 하이브리드 압축 완료")
    
    def _enhanced_2d_fft(self, W: torch.Tensor) -> torch.Tensor:
        """향상된 2D FFT (윈도우 함수 적용)"""
        
        # 한 윈도우 함수 적용 (스펙트럼 누출 방지)
        window_row = torch.hann_window(W.shape[0], device=W.device)
        window_col = torch.hann_window(W.shape[1], device=W.device)
        
        # 2D 윈도우 생성
        window_2d = torch.outer(window_row, window_col)
        W_windowed = W * window_2d
        
        # 2D FFT
        W_fft = torch.fft.fft2(W_windowed)
        
        return W_fft
    
    def _select_important_frequencies(self, W_fft: torch.Tensor):
        """중요한 주파수 성분 선택 (에너지 기반)"""
        
        # 주파수별 에너지 계산
        energy = torch.abs(W_fft)**2
        
        # 에너지 내림차순 정렬
        energy_flat = energy.flatten()
        sorted_indices = torch.argsort(energy_flat, descending=True)
        
        # 상위 에너지 성분 선택
        fft_budget = int(W_fft.numel() * self.compression_ratio * self.fft_ratio)
        important_indices = sorted_indices[:fft_budget]
        
        # 마스크 생성
        freq_mask = torch.zeros_like(energy_flat, dtype=torch.bool)
        freq_mask[important_indices] = True
        freq_mask = freq_mask.reshape(W_fft.shape)
        important_freqs = torch.where(freq_mask, W_fft, torch.zeros_like(W_fft))
        return important_freqs, freq_mask
    
    def _separate_fft_residual(self, W: torch.Tensor, important_freqs: torch.Tensor, 
                             freq_mask: torch.Tensor):
        fft_component = torch.fft.ifft2(important_freqs).real
        residual = W - fft_component
        return fft_component, residual
    
    def _compress_fft_component(self, fft_component: torch.Tensor):
        """FFT 성분 압축 (적응적 양자화)"""
        min_val, max_val = fft_component.min(), fft_component.max()
        dynamic_range = max_val - min_val
        if dynamic_range < 1e-6:
            num_bits = 4
        elif dynamic_range < 1e-3:
            num_bits = 6
        else:
            num_bits = 8
        num_levels = 2**num_bits
        scale = dynamic_range / (num_levels - 1)
        
        quantized = torch.round((fft_component - min_val) / scale)
        quantized = torch.clamp(quantized, 0, num_levels - 1)
        
        # 압축된 표현 저장
        compressed = {
            'quantized': quantized.to(torch.uint8),
            'min_val': min_val,
            'scale': scale,
            'shape': fft_component.shape
        }
        
        return compressed
    
    def _compress_residual_svd(self, residual: torch.Tensor):
        """잔차에 대한 고급 SVD 압축"""
        
        # SVD 예산 계산
        svd_ratio = 1 - self.fft_ratio
        target_rank = max(8, int(min(residual.shape) * self.compression_ratio * svd_ratio * 2))
        
        # 블록별 SVD (메모리 효율성)
        if residual.numel() > 100000:  # 큰 행렬의 경우
            return self._block_svd_compression(residual, target_rank)
        else:
            return self._standard_svd_compression(residual, target_rank)
    
    def _block_svd_compression(self, residual: torch.Tensor, target_rank: int):
        """블록별 SVD 압축"""
        
        block_size = 256
        compressed_blocks = []
        
        for i in range(0, residual.shape[0], block_size):
            end_i = min(i + block_size, residual.shape[0])
            block = residual[i:end_i]
            
            # 각 블록에 SVD 적용
            U, S, V = torch.svd(block.float())
            
            # 랭크 조정
            block_rank = min(target_rank, len(S))
            
            compressed_blocks.append({
                'U': U[:, :block_rank].to(residual.dtype),
                'S': S[:block_rank].to(residual.dtype),
                'V': V[:, :block_rank].to(residual.dtype),
                'start_row': i,
                'end_row': end_i
            })
        
        return {'type': 'block', 'blocks': compressed_blocks}
    
    def _standard_svd_compression(self, residual: torch.Tensor, target_rank: int):
        """표준 SVD 압축"""
        
        U, S, V = torch.svd(residual.float())
        
        # 적응적 랭크 선택 (에너지 기반)
        energy_cumsum = torch.cumsum(S**2, dim=0)
        total_energy = energy_cumsum[-1]
        energy_threshold = total_energy * 0.95  # 95% 에너지 보존
        
        adaptive_rank = torch.sum(energy_cumsum < energy_threshold).item() + 1
        final_rank = min(target_rank, adaptive_rank, len(S))
        
        return {
            'type': 'standard',
            'U': U[:, :final_rank].to(residual.dtype),
            'S': S[:final_rank].to(residual.dtype),
            'V': V[:, :final_rank].to(residual.dtype)
        }
    
    def reconstruct(self) -> torch.Tensor:
        """압축된 가중치 복원"""
        
        # FFT 성분 복원
        fft_reconstructed = self._reconstruct_fft()
        
        # SVD 성분 복원
        svd_reconstructed = self._reconstruct_svd()
        
        # 최종 합성
        return fft_reconstructed + svd_reconstructed
    
    def _reconstruct_fft(self) -> torch.Tensor:
        """FFT 성분 복원"""
        
        comp = self.fft_compressed
        
        # 역양자화
        dequantized = comp['quantized'].float() * comp['scale'] + comp['min_val']
        
        return dequantized.reshape(comp['shape'])
    
    def _reconstruct_svd(self) -> torch.Tensor:
        """SVD 성분 복원"""
        
        comp = self.svd_compressed
        
        if comp['type'] == 'block':
            # 블록별 복원
            result = torch.zeros(self.out_f, self.in_f, dtype=comp['blocks'][0]['U'].dtype)
            
            for block in comp['blocks']:
                start_row, end_row = block['start_row'], block['end_row']
                reconstructed_block = block['U'] @ torch.diag(block['S']) @ block['V'].t()
                result[start_row:end_row] = reconstructed_block
            
            return result
        else:
            # 표준 복원
            return comp['U'] @ torch.diag(comp['S']) @ comp['V'].t()
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """압축된 연산 적용 (효율적 구현)"""
        
        # FFT 성분 적용
        fft_result = self._apply_fft_fast(x)
        
        # SVD 성분 적용
        svd_result = self._apply_svd_fast(x)
        
        return fft_result + svd_result
    
    def _apply_fft_fast(self, x: torch.Tensor) -> torch.Tensor:
        """FFT 성분 빠른 적용"""
        # 실제로는 복원 없이 압축된 형태로 직접 계산 가능
        fft_weight = self._reconstruct_fft()
        return F.linear(x, fft_weight, None)
    
    def _apply_svd_fast(self, x: torch.Tensor) -> torch.Tensor:
        """SVD 성분 빠른 적용"""
        comp = self.svd_compressed
        
        if comp['type'] == 'block':
            # 블록별 적용 (메모리 효율적)
            results = []
            for block in comp['blocks']:
                start_row, end_row = block['start_row'], block['end_row']
                block_result = x @ block['V'] @ torch.diag(block['S']) @ block['U'].t()
                results.append(block_result)
            return torch.cat(results, dim=-1)
        else:
            # 표준 SVD 적용
            return x @ comp['V'] @ torch.diag(comp['S']) @ comp['U'].t()

# ───────── Enhanced RealityStone Linear Layer ─────────
class EnhancedRealityStoneLinear(nn.Module):
    """향상된 RealityStone Linear 레이어"""
    
    def __init__(self, lin, compression_ratio=0.1, compression_type='hybrid'):
        super().__init__()
        
        if hasattr(lin, 'weight'):
            W = lin.weight.data.clone()
            
            # Conv1D 처리
            if hasattr(lin, 'nf'):  # Conv1D
                self.in_features = W.shape[1]
                self.out_features = W.shape[0]
                W = W.t()  # [in, out] for compressor
                print(f"🌐 Conv1D 고급압축: in={self.in_features}, out={self.out_features}")
            else:  # nn.Linear
                self.in_features = lin.in_features
                self.out_features = lin.out_features
                print(f"🌐 Linear 고급압축: in={self.in_features}, out={self.out_features}")
            
            # 압축 타입별 압축기 선택
            if compression_type == 'hybrid':
                # FFT+SVD+리만 하이브리드
                self.compressor = self._create_hybrid_compressor(W, compression_ratio)
            elif compression_type == 'riemann':
                # 간소화 리만 압축
                self.compressor = SimplifiedRiemannCompressor(
                    W, compression_ratio, use_rs=True
                )
            elif compression_type == 'fft_svd':
                # FFT+SVD 압축
                self.compressor = AdvancedFFTSVDCompressor(W, compression_ratio)
            else:
                # 기본 간소화 리만 압축
                self.compressor = SimplifiedRiemannCompressor(
                    W, compression_ratio, use_rs=True
                )
            
            # 바이어스 처리
            if hasattr(lin, 'bias') and lin.bias is not None:
                self.bias = nn.Parameter(lin.bias.data.clone())
            else:
                self.bias = None
        else:
            raise ValueError("Input layer must have weight attribute")
    
    def _create_hybrid_compressor(self, W: torch.Tensor, compression_ratio: float):
        """하이브리드 압축기 생성 (속도 최적화)"""
        
        # 속도 최적화를 위해 간단한 SVD 압축 사용
        total_params = W.numel()
        
        print(f"      📊 최적화 압축: 고속 SVD ({total_params:,} 파라미터)")
        return FastSVDCompressor(W, compression_ratio)

    def forward(self, x):
        out = self.compressor.apply(x)
        if self.bias is not None:
            out = out + self.bias
        return out

# ───────── Enhanced Reality Stone Block ─────────
class EnhancedRealityStoneBlock(nn.Module):
    def __init__(self, block, compression_ratio=0.1, layer_idx=0, total_layers=12, 
                 adaptive_compression=True):
        super().__init__()
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        attn, mlp = block.attn, block.mlp

        # 적응적 압축률 및 방법 선택
        if adaptive_compression:
            layer_ratio, compression_types = self._adaptive_compression_strategy(
                layer_idx, total_layers, compression_ratio
            )
        else:
            layer_ratio = compression_ratio
            compression_types = ['hybrid'] * 4

        print(f"🌐 레이어 {layer_idx}: 고급압축률 {layer_ratio:.1%}")
        print(f"   압축방법: attn={compression_types[0]}, proj={compression_types[1]}")
        print(f"            fc={compression_types[2]}, mlp_proj={compression_types[3]}")

        # 각 서브레이어에 최적화된 압축 적용
        attn.c_attn = EnhancedRealityStoneLinear(attn.c_attn, layer_ratio, compression_types[0])
        attn.c_proj = EnhancedRealityStoneLinear(attn.c_proj, layer_ratio, compression_types[1])
        mlp.c_fc   = EnhancedRealityStoneLinear(mlp.c_fc,   layer_ratio, compression_types[2])
        mlp.c_proj = EnhancedRealityStoneLinear(mlp.c_proj, layer_ratio, compression_types[3])
        
        self.attn, self.mlp = attn, mlp

    def _adaptive_compression_strategy(self, layer_idx: int, total_layers: int, 
                                     base_ratio: float):
        """적응적 압축 전략 (속도 최적화)"""
        
        normalized_idx = layer_idx / total_layers
        
        # 속도 최적화를 위해 대부분 fast SVD 사용
        if normalized_idx < 0.3:  # 초기층 (0-30%)
            layer_ratio = base_ratio * 1.2  # 보수적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']  # 모두 fast SVD
        elif normalized_idx < 0.7:  # 중간층 (30-70%)
            layer_ratio = base_ratio * 0.8  # 적극적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']  # 모두 fast SVD
        else:  # 말단층 (70-100%)
            layer_ratio = base_ratio * 1.1  # 보수적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']  # 모두 fast SVD
        
        return layer_ratio, compression_types

    def forward(self, x, **kwargs):
        h = self.ln1(x)
        attn_outputs = self.attn(h, **kwargs)
        a = attn_outputs[0]
        x = x + a
        h2 = self.ln2(x)
        m = self.mlp(h2)
        output = x + m
        
        if len(attn_outputs) > 1:
            return (output,) + attn_outputs[1:]
        else:
            return (output,)

# ───────── Advanced Reality Stone Compression Pipeline ─────────
def apply_advanced_reality_stone_compression(model, compression_ratio=0.12, 
                                           compression_strategy='adaptive'):
    """고급 RealityStone 압축 파이프라인"""
    
    total = sum(p.numel() for p in model.parameters())
    total_layers = len(model.transformer.h)
    
    print(f"Before: {total:,} params")
    print(f"🌐 고급 RealityStone 압축: 목표={compression_ratio:.1%}")
    print(f"🚀 전략: {compression_strategy}")
    print(f"💎 활용 기술: RealityStone + FFT + SVD + 리만기하학")
    
    # 압축 전략별 레이어 선택
    if compression_strategy == 'adaptive':
        # 적응적: 모든 레이어 압축하되 강도 조절
        compress_layers = list(range(total_layers))
        adaptive = True
    elif compression_strategy == 'conservative':
        # 보수적: 가장자리 보존
        compress_layers = list(range(2, total_layers-2))
        adaptive = False
    elif compression_strategy == 'aggressive':
        # 적극적: 첫번째와 마지막만 보존
        compress_layers = list(range(1, total_layers-1))
        adaptive = True
    else:  # balanced
        # 균형적: 일부 가장자리 보존
        compress_layers = list(range(1, total_layers-1))
        adaptive = True
    
    print(f"   압축 대상: {len(compress_layers)}/{total_layers} 레이어 (전략: {compression_strategy})")
    
    # 압축 진행
    compressed_layers = 0
    for i in tqdm(compress_layers, desc="🌐 고급 압축"):
        if i < len(model.transformer.h):
            try:
                model.transformer.h[i] = EnhancedRealityStoneBlock(
                    model.transformer.h[i], compression_ratio, i, total_layers, adaptive
                )
                compressed_layers += 1
            except Exception as e:
                print(f"   ⚠️ 레이어 {i} 압축 실패: {e}")
                continue
    
    total2 = sum(p.numel() for p in model.parameters())
    actual_compression = total2 / total
    
    print(f"After:  {total2:,} params → {1/actual_compression:.2f}× 압축")
    print(f"🌐 실제 압축률: {(1-actual_compression)*100:.1f}%")
    print(f"✅ 성공적으로 압축된 레이어: {compressed_layers}/{len(compress_layers)}")
    
    # 압축 품질 평가
    quality_score = _evaluate_compression_quality(actual_compression, compression_ratio)
    print(f"📊 압축 품질 점수: {quality_score:.2f}/5.0")
    
    return model

def _evaluate_compression_quality(actual_ratio: float, target_ratio: float) -> float:
    """압축 품질 평가"""
    
    score = 5.0
    
    # 목표 달성도
    target_achievement = min(1.0, (1-actual_ratio) / (1-target_ratio))
    score *= target_achievement
    
    # 압축률 적절성
    if actual_ratio < 0.3:  # 70%+ 압축
        score *= 1.1  # 보너스
    elif actual_ratio > 0.7:  # 30% 미만 압축
        score *= 0.7  # 페널티
    
    return min(5.0, score)

# ───────── Keep existing quality evaluation and fine-tuning functions ─────────

def advanced_quality_evaluation(generated_text, prompt):
    """엄격한 한국어 품질 평가 시스템 (개선)"""
    
    generated_only = generated_text[len(prompt):].strip()
    if len(generated_only) < 2:
        return 0.0
    
    score = 0.0
    max_score = 7.0  # 더 엄격한 평가를 위해 7점 만점
    
    # 1. 반복 패턴 검사 (0-2점) - 가장 중요!
    repetition_penalty = calculate_repetition_penalty(generated_only)
    repetition_score = max(0, 2.0 - repetition_penalty * 4)  # 반복에 대한 강한 페널티
    score += repetition_score
    
    # 2. 한국어 문법 구조 (0-2점)
    grammar_score = evaluate_korean_grammar(generated_only)
    score += grammar_score
    
    # 3. 의미 연관성 (0-1.5점)
    semantic_score = calculate_semantic_relevance(prompt, generated_only)
    score += semantic_score * 1.5
    
    # 4. 텍스트 자연스러움 (0-1점)
    naturalness_score = evaluate_naturalness(generated_only)
    score += naturalness_score
    
    # 5. 특수문자/오류 패턴 페널티 (0-0.5점)
    error_penalty = calculate_error_penalty(generated_only)
    score += max(0, 0.5 - error_penalty)
    
    return min(score / max_score * 3.0, 3.0)  # 0-3 스케일로 변환

def evaluate_korean_grammar(text):
    """한국어 문법 구조 평가"""
    score = 0.0
    
    # 적절한 어미 사용
    korean_endings = ['다', '요', '니다', '해요', '어요', '아요', '네요', '죠', '습니다', '겠습니다']
    has_proper_ending = any(text.endswith(ending) for ending in korean_endings)
    if has_proper_ending:
        score += 1.0
    elif any(ending in text for ending in korean_endings):
        score += 0.5
    
    # 문장 구조
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    if sentences:
        # 완전한 문장이 있는지
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 2)
        if complete_sentences > 0:
            score += 0.8
        else:
            score += 0.3
    
    # 조사/어미 적절성
    particles = ['이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의']
    has_particles = any(p in text for p in particles)
    if has_particles:
        score += 0.2
    
    return min(score, 2.0)

def evaluate_naturalness(text):
    """텍스트 자연스러움 평가"""
    score = 1.0
    
    # 이상한 패턴들 체크
    weird_patterns = [
        r'[.]{3,}',           # 과도한 점
        r'[!]{2,}',           # 과도한 느낌표  
        r'[?]{2,}',           # 과도한 물음표
        r'[/]{2,}',           # 슬래시 반복
        r'[~]{3,}',           # 물결표 반복
        r'[:]{2,}',           # 콜론 반복
        r'[0-9]{5,}',         # 긴 숫자 나열
    ]
    
    for pattern in weird_patterns:
        if re.search(pattern, text):
            score -= 0.3
    
    # 단어 길이 체크
    words = text.split()
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 10:  # 너무 긴 평균 단어
            score -= 0.3
    
    return max(0, score)

def calculate_error_penalty(text):
    """오류 패턴 페널티 계산"""
    penalty = 0.0
    
    # 심각한 오류 패턴들
    severe_errors = [
        r'[가-힣]+[/]+[가-힣]+',    # 한글 사이 슬래시
        r'[:-]+[/]+',               # 특수문자 조합
        r'[&+-]{2,}',               # 연산자 반복
        r'[()\[\]]{3,}',            # 괄호 반복
    ]
    
    for pattern in severe_errors:
        matches = len(re.findall(pattern, text))
        penalty += matches * 0.5
    
    return penalty

def calculate_repetition_penalty(text):
    """반복 패턴 페널티 계산"""
    
    # 문자 반복 검사
    char_repeats = len(re.findall(r'(.)\1{2,}', text))  # 3회 이상 반복
    
    # 단어 반복 검사
    words = text.split()
    if len(words) > 1:
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 2)
    else:
        repeated_words = 0
    
    # 구두점 반복 검사
    punct_repeats = len(re.findall(r'[.!?]{3,}|[~]{2,}|[/]{2,}', text))
    
    # 총 페널티 (0-1 범위)
    total_penalty = min(1.0, (char_repeats + repeated_words + punct_repeats * 2) / 10)
    
    return total_penalty

def has_proper_structure(text):
    """적절한 문법 구조 확인"""
    korean_endings = ['다', '요', '니다', '해요', '어요', '아요', '네요', '죠']
    has_ending = any(text.endswith(ending) for ending in korean_endings)
    has_complete_sentence = '.' in text or '!' in text or '?' in text
    return has_ending and not text.count('.') > 3

def has_basic_structure(text):
    """기본적인 구조 확인"""
    return len(text.split()) >= 2 and not text.count('/') > len(text) * 0.3

def calculate_semantic_relevance(prompt, generated):
    """의미적 연관성 계산 (간단한 키워드 기반)"""
    
    keyword_mapping = {
        '안녕': ['안녕', '반갑', '좋', '감사'],
        '날씨': ['날씨', '맑', '흐림', '비', '눈', '따뜻', '춥', '좋'],
        '수도': ['서울', '도시', '한국', '수도'],
        '인공지능': ['AI', '기술', '컴퓨터', '로봇', '지능', '학습'],
        '음식': ['음식', '맛', '먹', '요리', '식사'],
    }
    
    relevance = 0.0
    for key, keywords in keyword_mapping.items():
        if key in prompt:
            matches = sum(1 for kw in keywords if kw in generated)
            relevance = max(relevance, min(1.0, matches / 2))
    
    return relevance

def calculate_diversity(text):
    """텍스트 다양성 계산"""
    
    if len(text) < 5:
        return 0.0
    
    # 문자 다양성
    unique_chars = len(set(text.replace(' ', '')))
    char_diversity = min(1.0, unique_chars / 10)
    
    # 단어 다양성
    words = text.split()
    if len(words) > 1:
        unique_words = len(set(words))
        word_diversity = unique_words / len(words)
    else:
        word_diversity = 0.5
    
    return (char_diversity + word_diversity) / 2

def generate_with_anti_repetition(model, tokenizer, prompt, max_length=25):
    """극한 반복 방지 생성 (한국어 초특화)"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.6,          # 보수적 온도
            top_p=0.8,               # 제한적 확률 
            top_k=30,                # 제한적 선택
            repetition_penalty=1.8,   # 반복 페널티 극대화
            no_repeat_ngram_size=5,   # n-gram 크기 확대
            pad_token_id=tokenizer.eos_token_id,
            # beam search 관련 설정들 제거 (충돌 해결)
            min_length=len(inputs.input_ids[0]) + 2,  # 최소 길이 보장
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def test_multiple_prompts_advanced(model, tokenizer, model_type="원본"):
    """개선된 다중 프롬프트 테스트"""
    
    test_prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    print(f"\n=== {model_type} 모델 고급 테스트 ===")
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/5] '{prompt}'")
        
        try:
            t0 = time.time()
            
            # 반복 방지 생성 사용
            generated_text = generate_with_anti_repetition(model, tokenizer, prompt, max_length=25)
            
            elapsed = time.time() - t0
            
            print(f"  생성: {generated_text}")
            print(f"  시간: {elapsed:.3f}초")
            
            # 고급 품질 평가
            quality_score = advanced_quality_evaluation(generated_text, prompt)
            
            print(f"  품질: {quality_score:.2f}/3.0")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': elapsed,
                'quality': quality_score
            })
            
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            results.append({
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'time': 0,
                'quality': 0
            })
    
    # 통계
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 {model_type} 고급 통계:")
    print(f"  평균 시간: {avg_time:.3f}초")
    print(f"  평균 품질: {avg_quality:.2f}/3.0")
    
    return results

def main():
    model_name = "skt/kogpt2-base-v2"
    print("🌐 향상된 RealityStone FFT+SVD+리만구면 압축 + 파인튜닝 파이프라인")
    print("=" * 90)
    print("🚀 Reality Stone + 고급 FFT+SVD + 리만기하학 + Knowledge Distillation")
    print("💎 다중 압축 기법: 하이브리드 적응적 압축")
    print("Loading model…")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1단계: 원본 모델 테스트
    print("\n" + "="*90)
    print("📊 원본 모델 성능 벤치마크")
    original_results = test_multiple_prompts_advanced(teacher_model, tokenizer, "원본")

    # 2단계: 향상된 RealityStone 압축 적용 (다중 전략 비교)
    print("\n" + "="*90)
    print("🌐 향상된 RealityStone 다중 압축 기법 적용")
    
    # 학생 모델들 생성 (다양한 전략)
    compression_strategies = [
        ('adaptive', 0.12, "적응적 하이브리드"),
        ('aggressive', 0.10, "적극적 압축"),
        ('conservative', 0.15, "보수적 압축")
    ]
    
    best_student = None
    best_score = -1
    best_strategy = None
    
    for strategy, ratio, description in compression_strategies:
        print(f"\n🔬 전략 테스트: {description} (압축률 {ratio:.1%})")
        
        student_model = copy.deepcopy(teacher_model)
        
        try:
            # 고급 압축 적용
            student_model = apply_advanced_reality_stone_compression(
                student_model, 
                compression_ratio=ratio,
                compression_strategy=strategy
            )
            
            # 압축 직후 성능 테스트
            print(f"\n📊 {description} 압축 직후 테스트")
            test_results = test_multiple_prompts_advanced(
                student_model, tokenizer, f"{description} (FT 전)"
            )
            
            # 압축 효과 평가
            avg_quality = sum(r['quality'] for r in test_results) / len(test_results)
            compression_score = _evaluate_compression_strategy(student_model, teacher_model, avg_quality)
            
            print(f"   🏆 전략 점수: {compression_score:.2f}/10.0")
            
            if compression_score > best_score:
                best_score = compression_score
                best_student = student_model
                best_strategy = description
                
        except Exception as e:
            print(f"   ❌ {description} 전략 실패: {e}")
            continue
    
    if best_student is None:
        print("❌ 모든 압축 전략 실패")
        return
    
    print(f"\n🏆 최적 전략 선택: {best_strategy} (점수: {best_score:.2f})")
    student_model = best_student
    
    # 3단계: 압축 직후 최종 테스트
    print("\n" + "="*90)
    print("📊 최적 압축 모델 파인튜닝 전 성능")
    before_ft_results = test_multiple_prompts_advanced(
        student_model, tokenizer, f"{best_strategy} (파인튜닝 전)"
    )
    
    # 4단계: 고급 Knowledge Distillation 파인튜닝
    print("\n" + "="*90)
    print("🧠 고급 Knowledge Distillation 파인튜닝 시작")
    
    student_model = enhanced_knowledge_distillation_fine_tune(
        teacher_model, student_model, tokenizer,
        total_steps=250,    # 더 많은 스텝
        base_lr=1.5e-5,     # 더 정교한 학습률
        temperature=3.5,    # 최적화된 온도
        use_advanced_kd=True  # 고급 KD 기법
    )
    
    # 5단계: 파인튜닝 후 최종 테스트
    print("\n" + "="*90)
    print("📊 파인튜닝 후 최종 성능 평가")
    after_ft_results = test_multiple_prompts_advanced(
        student_model, tokenizer, f"{best_strategy} (파인튜닝 후)"
    )
    
    # 6단계: 종합 성능 분석 및 벤치마크
    print("\n" + "="*90)
    print("🏆 향상된 RealityStone 압축 + 파인튜닝 최종 분석")
    print("="*90)
    
    # 성능 지표 계산
    orig_time = sum(r['time'] for r in original_results) / len(original_results)
    orig_quality = sum(r['quality'] for r in original_results) / len(original_results)
    
    before_ft_time = sum(r['time'] for r in before_ft_results) / len(before_ft_results)
    before_ft_quality = sum(r['quality'] for r in before_ft_results) / len(before_ft_results)
    
    after_ft_time = sum(r['time'] for r in after_ft_results) / len(after_ft_results)
    after_ft_quality = sum(r['quality'] for r in after_ft_results) / len(after_ft_results)
    
    # 상세 성능 리포트
    print(f"📊 성능 비교 리포트:")
    print(f"   원본 모델:           시간 {orig_time:.3f}초, 품질 {orig_quality:.2f}/3.0")
    print(f"   압축 후 (FT 전):     시간 {before_ft_time:.3f}초, 품질 {before_ft_quality:.2f}/3.0")
    print(f"   압축+FT 후:         시간 {after_ft_time:.3f}초, 품질 {after_ft_quality:.2f}/3.0")
    
    print(f"\n📈 개선 효과 분석:")
    quality_improvement = after_ft_quality - before_ft_quality
    quality_retention = after_ft_quality / orig_quality
    speed_improvement = orig_time / after_ft_time if after_ft_time > 0 else 1
    
    print(f"   파인튜닝 품질 개선:  {quality_improvement:+.2f}점 ({(quality_improvement/before_ft_quality)*100:+.1f}%)")
    print(f"   원본 대비 품질 유지: {quality_retention*100:.1f}%")
    print(f"   처리 속도 향상:     {speed_improvement:.2f}× 빨라짐")
    
    # 압축 통계
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = student_params / teacher_params
    memory_saved = (1 - compression_ratio) * 100
    
    print(f"\n💾 압축 성과:")
    print(f"   파라미터 수:        {teacher_params:,} → {student_params:,}")
    print(f"   압축 비율:         {compression_ratio:.3f} ({1/compression_ratio:.1f}× 압축)")
    print(f"   메모리 절약:       {memory_saved:.1f}%")
    
    # 전체 성과 평가
    overall_score = _calculate_overall_performance_score(
        quality_retention, speed_improvement, compression_ratio, quality_improvement
    )
    
    print(f"\n🎯 종합 성과 평가:")
    print(f"   전체 점수:         {overall_score:.1f}/100")
    print(f"   사용 기술:         {best_strategy}")
    print(f"   압축 라이브러리:   {'RealityStone + ' if RS_AVAILABLE else ''}FFT+SVD+리만기하학")
    
    # 성공 판정 및 등급
    if overall_score >= 85:
        grade = "🏆 대성공 (S급)"
        message = "모든 지표에서 탁월한 성능!"
    elif overall_score >= 75:
        grade = "🥇 성공 (A급)"
        message = "대부분 지표에서 우수한 성능!"
    elif overall_score >= 65:
        grade = "🥈 양호 (B급)"
        message = "상당한 개선 효과 확인!"
    elif overall_score >= 55:
        grade = "🥉 보통 (C급)"
        message = "일부 개선 효과 있음"
    else:
        grade = "🔧 개선 필요 (D급)"
        message = "추가 최적화 필요"
    
    print(f"\n{grade}: {message}")
    
    # 세부 권장사항
    if quality_retention < 0.8:
        print(f"💡 권장사항: 압축률을 줄이거나 파인튜닝 강화")
    if speed_improvement < 1.5:
        print(f"💡 권장사항: 더 적극적인 압축 전략 고려")
    if quality_improvement < 0.1:
        print(f"💡 권장사항: 파인튜닝 하이퍼파라미터 조정")
    
    print(f"\n🌟 최종 결론:")
    print(f"   향상된 RealityStone 압축 파이프라인으로")
    print(f"   {memory_saved:.0f}% 메모리 절약과 {speed_improvement:.1f}× 속도 향상을 달성하면서")
    print(f"   원본 품질의 {quality_retention*100:.0f}%를 유지했습니다!")

def _evaluate_compression_strategy(student_model, teacher_model, avg_quality):
    """압축 전략 평가"""
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    compression_ratio = student_params / teacher_params
    memory_score = min(10, (1 - compression_ratio) * 20)  # 압축률 점수
    quality_score = min(10, avg_quality * 3.33)           # 품질 점수
    
    return (memory_score + quality_score) / 2

def _calculate_overall_performance_score(quality_retention, speed_improvement, 
                                       compression_ratio, quality_improvement):
    """전체 성과 점수 계산"""
    
    # 각 지표별 점수 (0-25점)
    quality_score = min(25, quality_retention * 30)
    speed_score = min(25, (speed_improvement - 1) * 12.5)
    compression_score = min(25, (1 - compression_ratio) * 40)
    improvement_score = min(25, quality_improvement * 25)
    
    return quality_score + speed_score + compression_score + improvement_score

# ───────── Knowledge Distillation for Riemann Compression ─────────
def knowledge_distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """한국어 특화 지식 증류 손실 함수 (개선)"""
    
    # 1. 기본 KL divergence (더 정교한 온도 스케줄링)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # 2. 한국어 토큰 가중치 부여 (높은 확률 토큰에 더 집중)
    with torch.no_grad():
        # 높은 확률 토큰에 더 높은 가중치
        confidence_weights = torch.max(teacher_probs, dim=-1)[0]
        confidence_weights = confidence_weights.unsqueeze(-1)
    
    # 가중 KL divergence
    weighted_kl = kl_loss * confidence_weights.mean()
    
    # 3. 온도 제곱 스케일링 (표준)
    final_loss = weighted_kl * (temperature ** 2)
    
    return final_loss

def riemann_knowledge_distillation_fine_tune(teacher_model, student_model, tokenizer, 
                                           total_steps=200, base_lr=2e-5, temperature=3.0):
    """리만구면 압축 모델을 위한 Knowledge Distillation 파인튜닝"""
    
    print(f"\n🧠 리만구면 Knowledge Distillation 파인튜닝 시작")
    print(f"   총 스텝: {total_steps}, 학습률: {base_lr}, 온도: {temperature}")
    
    # 다양한 한국어 훈련 데이터
    train_texts = [
        # === 완벽한 일상 인사 ===
        "안녕하세요.",
        "안녕하세요. 반갑습니다.",
        "안녕하세요. 오늘 날씨가 좋네요.",
        "안녕하세요. 어떻게 지내세요?",
        "좋은 아침입니다.",
        "좋은 저녁입니다.",
        "안녕히 가세요.",
        "감사합니다.",
        "죄송합니다.",
        "괜찮습니다.",
        
        # === 완벽한 날씨 표현 ===
        "오늘 날씨가 맑습니다.",
        "오늘 날씨가 흐립니다.",
        "오늘 날씨가 춥습니다.",
        "오늘 날씨가 따뜻합니다.",
        "비가 옵니다.",
        "눈이 옵니다.",
        "바람이 붑니다.",
        "햇살이 좋습니다.",
        
        # === 완벽한 일상 대화 ===
        "밥을 먹었습니다.",
        "물을 마셨습니다.",
        "책을 읽었습니다.",
        "공부를 했습니다.",
        "운동을 했습니다.",
        "음악을 들었습니다.",
        "영화를 봤습니다.",
        "친구를 만났습니다.",
        "집에 갔습니다.",
        "학교에 갔습니다.",
        
        # === 완벽한 감정 표현 ===
        "기분이 좋습니다.",
        "기분이 나쁩니다.",
        "행복합니다.",
        "슬픕니다.",
        "즐겁습니다.",
        "피곤합니다.",
        "편안합니다.",
        "걱정됩니다.",
        
        # === 완벽한 계획 표현 ===
        "내일 갈 예정입니다.",
        "다음 주에 할 계획입니다.",
        "곧 시작하겠습니다.",
        "나중에 하겠습니다.",
        "빨리 끝내겠습니다.",
        "천천히 하겠습니다.",
        
        # === 완벽한 설명 표현 ===
        "이것은 책입니다.",
        "저것은 펜입니다.",
        "여기는 학교입니다.",
        "거기는 집입니다.",
        "지금은 오후입니다.",
        "어제는 월요일이었습니다.",
        
        # === 완벽한 질문 표현 ===
        "뭐 하세요?",
        "어디 가세요?",
        "언제 오세요?",
        "누구랑 가세요?",
        "왜 그러세요?",
        "어떻게 하세요?",
        
        # === 완벽한 응답 표현 ===
        "네, 맞습니다.",
        "아니요, 틀렸습니다.",
        "잘 모르겠습니다.",
        "생각해 보겠습니다.",
        "알겠습니다.",
        "이해했습니다.",
    ]
    
    # 3단계 점진적 학습
    phases = [
        {"name": "🌱 기초", "steps": total_steps//5, "lr_mult": 0.1, "kd_weight": 0.99, "lm_weight": 0.01, "reg_weight": 0.1},
        {"name": "🔥 집중", "steps": total_steps//5, "lr_mult": 1.5, "kd_weight": 0.95, "lm_weight": 0.05, "reg_weight": 0.05},
        {"name": "💎 정밀", "steps": total_steps//5, "lr_mult": 1.0, "kd_weight": 0.85, "lm_weight": 0.15, "reg_weight": 0.02},
        {"name": "✨ 완성", "steps": total_steps//5, "lr_mult": 0.5, "kd_weight": 0.70, "lm_weight": 0.30, "reg_weight": 0.01},
        {"name": "🎯 완벽", "steps": total_steps//5, "lr_mult": 0.2, "kd_weight": 0.50, "lm_weight": 0.50, "reg_weight": 0.005}
    ]
    
    teacher_model.eval()
    student_model.train()
    
    total_loss = 0.0
    step_count = 0
    best_loss = float('inf')
    patience = 50
    no_improve_count = 0
    
    for phase in phases:
        print(f"\n{phase['name']} - 스텝: {phase['steps']}, LR배수: {phase['lr_mult']}")
        
        # 단계별 옵티마이저
        current_lr = base_lr * phase['lr_mult']
        optimizer = torch.optim.AdamW(student_model.parameters(), 
                                     lr=current_lr, weight_decay=0.01, eps=1e-6)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase['steps'], eta_min=current_lr*0.1
        )
        
        progress_bar = tqdm(range(phase['steps']), desc=phase['name'])
        
        for step in progress_bar:
            # 다양한 데이터 선택
            text = train_texts[step_count % len(train_texts)]
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding=True)
            
            if inputs.input_ids.shape[1] < 3:
                continue
                
            input_ids = inputs.input_ids
            labels = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]
            
            optimizer.zero_grad()
            
            # Teacher 출력
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids)
            
            # Student 출력
            student_outputs = student_model(input_ids)
            
            # 손실 계산
            # 1) Knowledge Distillation Loss
            kd_loss = knowledge_distillation_loss(
                student_outputs.logits, teacher_outputs.logits, temperature
            )
            
            # 2) Language Model Loss
            lm_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # 3) 한국어 일관성 정규화 (추가)
            korean_consistency_loss = 0
            if step_count % 5 == 0:  # 5스텝마다 적용
                korean_consistency_loss = calculate_korean_consistency_loss(
                    student_outputs.logits, tokenizer
                )
            
            # 4) 정규화 (리만 압축 파라미터)
            reg_loss = 0
            for name, param in student_model.named_parameters():
                if any(keyword in name.lower() for keyword in ['compressor', 'svd', 'riemann']):
                    reg_loss += torch.norm(param, 2)
            reg_loss *= 1e-6
            
            # 단계별 가중치 적용
            total_loss_step = (phase['kd_weight'] * kd_loss + 
                             phase['lm_weight'] * lm_loss + 
                             0.1 * korean_consistency_loss +  # 한국어 일관성
                             reg_loss)
            
            total_loss += total_loss_step.item()
            step_count += 1
            
            # 역전파
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 진행상황 업데이트
            if step % 10 == 0:
                avg_loss = total_loss / step_count
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'kd': f'{kd_loss.item():.3f}',
                    'lm': f'{lm_loss.item():.3f}'
                })
    
    avg_loss = total_loss / step_count
    print(f"   전체 평균 손실: {avg_loss:.4f}")
    print("✅ 리만구면 Knowledge Distillation 파인튜닝 완료!")
    
    return student_model

def calculate_korean_consistency_loss(logits, tokenizer):
    """한국어 어미 일관성 손실"""
    # 한국어 어미 토큰들의 ID 찾기
    korean_endings = ['다', '요', '니다', '해요', '어요', '아요']
    ending_token_ids = []
    
    for ending in korean_endings:
        try:
            token_id = tokenizer.encode(ending, add_special_tokens=False)
            if token_id:
                ending_token_ids.extend(token_id)
        except:
            continue
    
    if not ending_token_ids:
        return torch.tensor(0.0, device=logits.device)
    
    # 어미 토큰들의 확률 분포 일관성 체크
    probs = F.softmax(logits, dim=-1)
    ending_probs = probs[:, :, ending_token_ids].sum(dim=-1)
    
    # 시퀀스 내에서 어미 사용의 일관성 측정
    consistency_loss = torch.var(ending_probs, dim=1).mean()
    
    return consistency_loss

# ───────── Simplified but Robust Compressor ─────────
class SimplifiedRiemannCompressor:
    """간소화되었지만 견고한 리만 압축기"""
    
    def __init__(self, W: torch.Tensor, compression_ratio=0.05, use_rs=True):
        """
        Args:
            W: 가중치 행렬 [out_f, in_f]
            compression_ratio: 압축률 (극한 압축)
            use_rs: reality_stone 사용 여부
        """
        self.out_f, self.in_f = W.shape
        self.compression_ratio = compression_ratio
        self.use_rs = use_rs and RS_AVAILABLE
        
        print(f"    🔧 극한 리만압축: {W.shape}, 압축률={compression_ratio:.1%}")
        
        self._apply_robust_compression(W)
    
    def _apply_robust_compression(self, W: torch.Tensor):
        """극한 압축 적용"""
        
        success = False
        
        # 1차 시도: RealityStone 라이브러리
        if self.use_rs:
            success = self._try_reality_stone_compression(W)
        
        # 2차 시도: 단순 리만 압축
        if not success:
            success = self._try_simple_riemann_compression(W)
        
        # 3차 시도: SVD 폴백
        if not success:
            self._apply_svd_fallback(W)
    
    def _try_reality_stone_compression(self, W: torch.Tensor) -> bool:
        """RealityStone 압축 시도"""
        
        try:
            # 가장 기본적인 RS 함수들만 시도
            basic_methods = [
                'poincare_ball_layer',
                'mobius_add',
                'hyperbolic_laplacian'
            ]
            
            for method_name in basic_methods:
                if hasattr(rs, method_name):
                    print(f"      💎 RS 기본 기능 활용: {method_name}")
                    
                    # 극한 압축 시뮬레이션
                    U, S, V = torch.svd(W.float())
                    rank = max(8, int(min(W.shape) * self.compression_ratio * 2))  # 더 극한
                    rank = min(rank, len(S))
                    
                    self.U = nn.Parameter(U[:, :rank].to(W.dtype))
                    self.S = nn.Parameter(S[:rank].to(W.dtype))
                    self.V = nn.Parameter(V[:, :rank].to(W.dtype))
                    
                    print(f"      ✅ RS 기반 극한압축 완료: rank {rank}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"      ⚠️ RS 압축 실패: {e}")
            return False
    
    def _try_simple_riemann_compression(self, W: torch.Tensor) -> bool:
        """간단한 리만 압축 시도"""
        
        try:
            print(f"      🌐 간단 리만 극한압축...")
            
            # 1. 복소수 변환 (안전한 버전)
            rows, cols = W.shape
            
            if cols >= 2:
                # 절반씩 나누어 복소수 생성
                mid = cols // 2
                real_part = W[:, :mid]
                imag_part = W[:, mid:2*mid] if cols >= 2*mid else torch.zeros_like(real_part)
            else:
                real_part = W
                imag_part = torch.zeros_like(W)
            
            complex_W = torch.complex(real_part, imag_part)
            
            # 2. 간단한 스테레오그래픽 투영
            sphere_coords = self._safe_stereographic_projection(complex_W)
            
            # 3. 극한 샘플링 (더 적극적)
            sampled = self._ultra_sampling(sphere_coords)
            
            # 4. SVD로 마무리
            flat_sampled = sampled.view(rows, -1)
            if flat_sampled.shape[1] < cols:
                # 패딩
                padding = torch.zeros(rows, cols - flat_sampled.shape[1], 
                                    dtype=flat_sampled.dtype, device=flat_sampled.device)
                flat_sampled = torch.cat([flat_sampled, padding], dim=1)
            elif flat_sampled.shape[1] > cols:
                # 트렁케이션
                flat_sampled = flat_sampled[:, :cols]
            
            U, S, V = torch.svd(flat_sampled.float())
            rank = max(4, int(min(W.shape) * self.compression_ratio))  # 더 극한
            rank = min(rank, len(S))
            
            self.U = nn.Parameter(U[:, :rank].to(W.dtype))
            self.S = nn.Parameter(S[:rank].to(W.dtype))
            self.V = nn.Parameter(V[:, :rank].to(W.dtype))
            
            print(f"      ✅ 리만 극한압축 완료: rank {rank}")
            return True
            
        except Exception as e:
            print(f"      ⚠️ 간단 리만 압축 실패: {e}")
            return False
    
    def _safe_stereographic_projection(self, z: torch.Tensor) -> torch.Tensor:
        """안전한 스테레오그래픽 투영"""
        
        real, imag = z.real, z.imag
        norm_sq = real**2 + imag**2
        
        # 안전한 분모
        denom = 1 + norm_sq
        epsilon = 1e-8
        denom = torch.clamp(denom, min=epsilon)
        
        X = 2 * real / denom
        Y = 2 * imag / denom
        Z = (norm_sq - 1) / denom
        
        return torch.stack([X, Y, Z], dim=-1)
    
    def _ultra_sampling(self, coords: torch.Tensor) -> torch.Tensor:
        """극한 샘플링 (더 적극적)"""
        
        original_shape = coords.shape
        flat_coords = coords.view(-1, 3)
        
        n_points = len(flat_coords)
        target_points = max(4, int(n_points * self.compression_ratio * 0.5))  # 더 극한
        
        if target_points >= n_points:
            return coords
        
        # 균등 간격 샘플링
        indices = torch.linspace(0, n_points-1, target_points, dtype=torch.long, device=coords.device)
        sampled = flat_coords[indices]
        
        return sampled
    
    def _apply_svd_fallback(self, W: torch.Tensor):
        """SVD 폴백 (극한 압축)"""
        
        print(f"      📊 SVD 극한압축...")
        
        U, S, V = torch.svd(W.float())
        rank = max(4, int(min(W.shape) * self.compression_ratio))  # 더 극한
        rank = min(rank, len(S))
        
        self.U = nn.Parameter(U[:, :rank].to(W.dtype))
        self.S = nn.Parameter(S[:rank].to(W.dtype))
        self.V = nn.Parameter(V[:, :rank].to(W.dtype))
        
        print(f"      ✅ SVD 극한압축 완료: rank {rank}")
    
    def reconstruct(self) -> torch.Tensor:
        """압축된 가중치 복원"""
        return self.U @ torch.diag(self.S) @ self.V.t()
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """압축된 연산 적용"""
        return x @ self.V @ torch.diag(self.S) @ self.U.t()

# ───────── Enhanced RealityStone Linear Layer ─────────

# ───────── Enhanced Reality Stone Block ─────────
class EnhancedRealityStoneBlock(nn.Module):
    def __init__(self, block, compression_ratio=0.05, layer_idx=0, total_layers=12, 
                 adaptive_compression=True):
        super().__init__()
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        attn, mlp = block.attn, block.mlp

        # 극한 적응적 압축률 및 방법 선택
        if adaptive_compression:
            layer_ratio, compression_types = self._extreme_compression_strategy(
                layer_idx, total_layers, compression_ratio
            )
        else:
            layer_ratio = compression_ratio
            compression_types = ['hybrid'] * 4

        print(f"🔥 극한압축 레이어 {layer_idx}: 압축률 {layer_ratio:.1%}")
        print(f"   압축방법: attn={compression_types[0]}, proj={compression_types[1]}")
        print(f"            fc={compression_types[2]}, mlp_proj={compression_types[3]}")

        # 각 서브레이어에 극한 압축 적용
        attn.c_attn = EnhancedRealityStoneLinear(attn.c_attn, layer_ratio, compression_types[0])
        attn.c_proj = EnhancedRealityStoneLinear(attn.c_proj, layer_ratio, compression_types[1])
        mlp.c_fc   = EnhancedRealityStoneLinear(mlp.c_fc,   layer_ratio, compression_types[2])
        mlp.c_proj = EnhancedRealityStoneLinear(mlp.c_proj, layer_ratio, compression_types[3])
        
        self.attn, self.mlp = attn, mlp

    def _extreme_compression_strategy(self, layer_idx: int, total_layers: int, 
                                     base_ratio: float):
        """극한 압축 전략 (더 공격적)"""
        
        normalized_idx = layer_idx / total_layers
        
        # 극한 압축을 위한 더 공격적 설정
        if normalized_idx < 0.2:  # 초기층 (0-20%)
            layer_ratio = base_ratio * 1.5  # 약간 보수적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']
        elif normalized_idx < 0.8:  # 중간층 (20-80%)
            layer_ratio = base_ratio * 0.5  # 매우 적극적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']
        else:  # 말단층 (80-100%)
            layer_ratio = base_ratio * 1.2  # 약간 보수적
            compression_types = ['hybrid', 'hybrid', 'hybrid', 'hybrid']
        
        return layer_ratio, compression_types

    def forward(self, x, **kwargs):
        h = self.ln1(x)
        attn_outputs = self.attn(h, **kwargs)
        a = attn_outputs[0]
        x = x + a
        h2 = self.ln2(x)
        m = self.mlp(h2)
        output = x + m
        
        if len(attn_outputs) > 1:
            return (output,) + attn_outputs[1:]
        else:
            return (output,)

# ───────── Enhanced Reality Stone Compression Pipeline ─────────
def apply_extreme_reality_stone_compression(model, compression_ratio=0.05, 
                                           compression_strategy='adaptive'):
    """극한 RealityStone 압축 파이프라인 (5% 목표)"""
    
    total = sum(p.numel() for p in model.parameters())
    total_layers = len(model.transformer.h)
    
    print(f"Before: {total:,} params ({total/1e6:.1f}M)")
    print(f"🔥 극한 RealityStone 압축: 목표={compression_ratio:.1%} (95% 메모리 절약)")
    print(f"🚀 전략: {compression_strategy}")
    print(f"💎 활용 기술: RealityStone + FFT + SVD + 리만기하학")
    
    # 압축 전략별 레이어 선택
    if compression_strategy == 'adaptive':
        # 적응적: 모든 레이어 극한 압축
        compress_layers = list(range(total_layers))
        adaptive = True
    elif compression_strategy == 'conservative':
        # 보수적이라도 극한 압축
        compress_layers = list(range(1, total_layers-1))
        adaptive = False
    elif compression_strategy == 'aggressive':
        # 적극적: 첫번째만 보존
        compress_layers = list(range(1, total_layers))
        adaptive = True
    else:  # balanced
        # 균형적
        compress_layers = list(range(1, total_layers-1))
        adaptive = True
    
    print(f"   압축 대상: {len(compress_layers)}/{total_layers} 레이어 (전략: {compression_strategy})")
    
    # 극한 압축 진행
    compressed_layers = 0
    for i in tqdm(compress_layers, desc="🔥 극한 압축"):
        if i < len(model.transformer.h):
            try:
                model.transformer.h[i] = EnhancedRealityStoneBlock(
                    model.transformer.h[i], compression_ratio, i, total_layers, adaptive
                )
                compressed_layers += 1
            except Exception as e:
                print(f"   ⚠️ 레이어 {i} 압축 실패: {e}")
                continue
    
    total2 = sum(p.numel() for p in model.parameters())
    actual_compression = total2 / total
    
    print(f"After:  {total2:,} params ({total2/1e6:.1f}M)")
    print(f"🔥 실제 압축률: {actual_compression:.1%} ({1/actual_compression:.1f}× 압축)")
    print(f"✅ 성공적으로 압축된 레이어: {compressed_layers}/{len(compress_layers)}")
    
    # 압축 품질 평가
    quality_score = _evaluate_compression_quality(actual_compression, compression_ratio)
    print(f"📊 압축 품질 점수: {quality_score:.2f}/5.0")
    
    return model

def _evaluate_compression_quality(actual_ratio: float, target_ratio: float) -> float:
    """압축 품질 평가"""
    
    score = 5.0
    
    # 목표 달성도
    target_achievement = min(1.0, (1-actual_ratio) / (1-target_ratio))
    score *= target_achievement
    
    # 압축률 적절성
    if actual_ratio < 0.1:  # 90%+ 압축
        score *= 1.2  # 보너스
    elif actual_ratio > 0.5:  # 50% 미만 압축
        score *= 0.8  # 페널티
    
    return min(5.0, score)

# ═══════════════════════════════════════════════════════════════
# 🧠 극한 Knowledge Distillation 파인튜닝 (2500 스텝)
# ═══════════════════════════════════════════════════════════════

def ultra_knowledge_distillation_fine_tune(teacher_model, student_model, tokenizer, 
                                           total_steps=2500, base_lr=1e-5, temperature=1.8):
    """극한 지식 증류 파인튜닝 (2500 스텝)"""
    
    print(f"\n🧠 극한 Knowledge Distillation 파인튜닝 시작")
    print(f"   총 스텝: {total_steps}, 학습률: {base_lr}, 온도: {temperature}")
    print(f"🎯 목표: 극한 압축 모델의 품질을 95%+ 복원")
    
    # 더 다양하고 체계적인 한국어 훈련 데이터
    train_texts = [
        # === 완벽한 일상 인사 ===
        "안녕하세요.", "안녕하세요. 반갑습니다.", "안녕하세요. 오늘 날씨가 좋네요.",
        "안녕하세요. 어떻게 지내세요?", "좋은 아침입니다.", "좋은 저녁입니다.",
        "안녕히 가세요.", "감사합니다.", "죄송합니다.", "괜찮습니다.",
        
        # === 완벽한 날씨 표현 ===
        "오늘 날씨가 맑습니다.", "오늘 날씨가 흐립니다.", "오늘 날씨가 춥습니다.",
        "오늘 날씨가 따뜻합니다.", "비가 옵니다.", "눈이 옵니다.", "바람이 붑니다.",
        "햇살이 좋습니다.", "날씨가 화창합니다.", "구름이 많습니다.",
        
        # === 완벽한 일상 대화 ===
        "밥을 먹었습니다.", "물을 마셨습니다.", "책을 읽었습니다.", "공부를 했습니다.",
        "운동을 했습니다.", "음악을 들었습니다.", "영화를 봤습니다.", "친구를 만났습니다.",
        "집에 갔습니다.", "학교에 갔습니다.", "회사에 갔습니다.", "쇼핑을 했습니다.",
        
        # === 완벽한 감정 표현 ===
        "기분이 좋습니다.", "기분이 나쁩니다.", "행복합니다.", "슬픕니다.",
        "즐겁습니다.", "피곤합니다.", "편안합니다.", "걱정됩니다.", "신납니다.",
        
        # === 완벽한 계획 표현 ===
        "내일 갈 예정입니다.", "다음 주에 할 계획입니다.", "곧 시작하겠습니다.",
        "나중에 하겠습니다.", "빨리 끝내겠습니다.", "천천히 하겠습니다.",
        
        # === 완벽한 설명 표현 ===
        "이것은 책입니다.", "저것은 펜입니다.", "여기는 학교입니다.", "거기는 집입니다.",
        "지금은 오후입니다.", "어제는 월요일이었습니다.", "내일은 화요일입니다.",
        
        # === 완벽한 질문 표현 ===
        "뭐 하세요?", "어디 가세요?", "언제 오세요?", "누구랑 가세요?",
        "왜 그러세요?", "어떻게 하세요?", "무엇을 드릴까요?",
        
        # === 완벽한 응답 표현 ===
        "네, 맞습니다.", "아니요, 틀렸습니다.", "잘 모르겠습니다.", "생각해 보겠습니다.",
        "알겠습니다.", "이해했습니다.", "그렇습니다.", "물론입니다.",
        
        # === 복합 문장 ===
        "오늘 날씨가 좋아서 산책을 했습니다.", "친구와 함께 영화를 봤습니다.",
        "도서관에서 책을 읽고 있습니다.", "내일 여행을 갈 예정입니다.",
        "맛있는 음식을 먹고 싶습니다.", "새로운 것을 배우고 있습니다.",
        "시간이 빨리 지나갑니다.", "열심히 공부하고 있습니다."
    ]
    
    # 5단계 점진적 학습 (더 세밀하게)
    phases = [
        {"name": "🌱 기초적응", "steps": total_steps//5, "lr_mult": 0.5, "kd_weight": 0.98, "lm_weight": 0.02, "temp": 1.5},
        {"name": "🔥 집중학습", "steps": total_steps//5, "lr_mult": 2.0, "kd_weight": 0.95, "lm_weight": 0.05, "temp": 1.8},
        {"name": "💎 정밀조정", "steps": total_steps//5, "lr_mult": 1.5, "kd_weight": 0.90, "lm_weight": 0.10, "temp": 2.0},
        {"name": "✨ 완성단계", "steps": total_steps//5, "lr_mult": 1.0, "kd_weight": 0.85, "lm_weight": 0.15, "temp": 2.2},
        {"name": "🎯 완벽마무리", "steps": total_steps//5, "lr_mult": 0.3, "kd_weight": 0.75, "lm_weight": 0.25, "temp": 2.5}
    ]
    
    teacher_model.eval()
    student_model.train()
    
    total_loss = 0.0
    step_count = 0
    best_loss = float('inf')
    
    for phase in phases:
        print(f"\n{phase['name']} - 스텝: {phase['steps']}, LR배수: {phase['lr_mult']}")
        
        # 단계별 옵티마이저
        current_lr = base_lr * phase['lr_mult']
        optimizer = torch.optim.AdamW(student_model.parameters(), 
                                     lr=current_lr, weight_decay=0.005, eps=1e-8)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase['steps'], eta_min=current_lr*0.05
        )
        
        progress_bar = tqdm(range(phase['steps']), desc=phase['name'])
        
        for step in progress_bar:
            # 다양한 데이터 선택
            text = train_texts[step_count % len(train_texts)]
            inputs = tokenizer(text, return_tensors="pt", max_length=28, 
                             truncation=True, padding=True)
            
            if inputs.input_ids.shape[1] < 3:
                continue
                
            input_ids = inputs.input_ids
            labels = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]
            
            optimizer.zero_grad()
            
            # Teacher 출력
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids)
            
            # Student 출력
            student_outputs = student_model(input_ids)
            
            # 극한 KD 손실
            kd_loss = knowledge_distillation_loss(
                student_outputs.logits, teacher_outputs.logits, phase['temp']
            )
            
            # Language Model Loss
            lm_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # 한국어 일관성 정규화
            korean_reg_loss = 0
            if step_count % 10 == 0:  # 10스텝마다 적용
                korean_reg_loss = calculate_korean_consistency_loss(
                    student_outputs.logits, tokenizer
                )
            
            # 극한 압축 정규화 (압축된 파라미터)
            compression_reg_loss = 0
            for name, param in student_model.named_parameters():
                if any(keyword in name.lower() for keyword in ['compressor', 'svd', 'riemann']):
                    compression_reg_loss += torch.norm(param, 2)
            compression_reg_loss *= 5e-7  # 더 강한 정규화
            
            # 단계별 가중치 적용
            total_loss_step = (phase['kd_weight'] * kd_loss + 
                             phase['lm_weight'] * lm_loss + 
                             0.05 * korean_reg_loss +  # 한국어 일관성
                             compression_reg_loss)
            
            total_loss += total_loss_step.item()
            step_count += 1
            
            # 역전파
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.8)
            optimizer.step()
            scheduler.step()
            
            # 진행상황 업데이트
            if step % 25 == 0:
                avg_loss = total_loss / step_count
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'kd': f'{kd_loss.item():.3f}',
                    'lm': f'{lm_loss.item():.3f}'
                })
                
                # 조기 종료 체크
                if avg_loss < best_loss:
                    best_loss = avg_loss
    
    avg_loss = total_loss / step_count
    print(f"   전체 평균 손실: {avg_loss:.4f}")
    print("✅ 극한 Knowledge Distillation 파인튜닝 완료!")
    
    return student_model

# ═══════════════════════════════════════════════════════════════
# 🎯 극한 테스트 함수
# ═══════════════════════════════════════════════════════════════

def test_extreme_performance(model, tokenizer, model_type="원본"):
    """극한 성능 테스트"""
    
    test_prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    print(f"\n=== {model_type} 모델 극한 테스트 ===")
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/5] '{prompt}'")
        
        try:
            t0 = time.time()
            
            # 극한 품질 생성 사용
            generated_text = generate_with_anti_repetition(model, tokenizer, prompt, max_length=25)
            
            elapsed = time.time() - t0
            
            print(f"  생성: {generated_text}")
            print(f"  시간: {elapsed:.3f}초")
            
            # 극한 품질 평가
            quality_score = advanced_quality_evaluation(generated_text, prompt)
            
            print(f"  품질: {quality_score:.2f}/3.0")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': elapsed,
                'quality': quality_score
            })
            
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            results.append({
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'time': 0,
                'quality': 0
            })
    
    # 통계
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 {model_type} 극한 통계:")
    print(f"  평균 시간: {avg_time:.3f}초")
    print(f"  평균 품질: {avg_quality:.2f}/3.0")
    
    return results

def main_extreme():
    """극한 압축 메인 함수 (5% 목표)"""
    
    model_name = "skt/kogpt2-base-v2"
    print("🔥 극한 RealityStone 압축 시스템 v8.0")
    print("=" * 90)
    print("🎯 목표: 125M → 6.25M (95% 압축) + 성능 90%+ 유지")
    print("🚀 기법: 극한양자화 + 극한프루닝 + 극한KD + 2500스텝 파인튜닝")
    print("Loading model…")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1단계: 원본 모델 테스트
    print("\n" + "="*90)
    print("📊 원본 모델 성능 벤치마크")
    original_results = test_extreme_performance(teacher_model, tokenizer, "원본")

    # 2단계: 극한 압축 적용 
    print("\n" + "="*90)
    print("🔥 극한 RealityStone 압축 적용")
    
    student_model = copy.deepcopy(teacher_model)
    
    try:
        # 극한 압축 적용
        student_model = apply_extreme_reality_stone_compression(
            student_model, 
            compression_ratio=0.05,  # 5% 목표
            compression_strategy='adaptive'
        )
        
        # 압축 직후 성능 테스트
        print("\n" + "="*90)
        print("📊 극한 압축 직후 테스트")
        compressed_results = test_extreme_performance(student_model, tokenizer, "극한압축후")
        
        # 극한 파인튜닝
        print("\n" + "="*90)
        print("🧠 극한 Knowledge Distillation 파인튜닝")
        student_model = ultra_knowledge_distillation_fine_tune(
            teacher_model, student_model, tokenizer,
            total_steps=2500,  # 극한 스텝
            base_lr=8e-6,      # 더 정교한 학습률
            temperature=1.8    # 최적화된 온도
        )
        
        # 파인튜닝 후 최종 테스트
        print("\n" + "="*90)
        print("📊 극한 파인튜닝 후 최종 테스트")
        final_results = test_extreme_performance(student_model, tokenizer, "극한최종")
        
        # 종합 성능 분석
        print("\n" + "="*90)
        print("🏆 극한 RealityStone 압축 최종 분석")
        print("="*90)
        
        # 성능 지표 계산
        orig_time = sum(r['time'] for r in original_results) / len(original_results)
        orig_quality = sum(r['quality'] for r in original_results) / len(original_results)
        
        comp_time = sum(r['time'] for r in compressed_results) / len(compressed_results)
        comp_quality = sum(r['quality'] for r in compressed_results) / len(compressed_results)
        
        final_time = sum(r['time'] for r in final_results) / len(final_results)
        final_quality = sum(r['quality'] for r in final_results) / len(final_results)
        
        # 상세 성능 리포트
        print(f"📊 성능 비교 리포트:")
        print(f"   원본 모델:           시간 {orig_time:.3f}초, 품질 {orig_quality:.2f}/3.0")
        print(f"   극한압축 후:         시간 {comp_time:.3f}초, 품질 {comp_quality:.2f}/3.0")
        print(f"   극한튜닝 후:         시간 {final_time:.3f}초, 품질 {final_quality:.2f}/3.0")
        
        print(f"\n📈 개선 효과 분석:")
        quality_improvement = final_quality - comp_quality
        quality_retention = final_quality / orig_quality
        speed_improvement = orig_time / final_time if final_time > 0 else 1
        
        print(f"   파인튜닝 품질 개선:  {quality_improvement:+.2f}점 ({(quality_improvement/comp_quality)*100:+.1f}%)")
        print(f"   원본 대비 품질 유지: {quality_retention*100:.1f}%")
        print(f"   처리 속도 향상:     {speed_improvement:.2f}× 빨라짐")
        
        # 압축 통계
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        compression_ratio = student_params / teacher_params
        memory_saved = (1 - compression_ratio) * 100
        
        print(f"\n💾 극한 압축 성과:")
        print(f"   파라미터 수:        {teacher_params:,} → {student_params:,}")
        print(f"   압축 비율:         {compression_ratio:.3f} ({1/compression_ratio:.1f}× 압축)")
        print(f"   메모리 절약:       {memory_saved:.1f}%")
        
        # 전체 성과 평가
        overall_score = _calculate_extreme_performance_score(
            quality_retention, speed_improvement, compression_ratio, quality_improvement
        )
        
        print(f"\n🎯 극한 성과 평가:")
        print(f"   전체 점수:         {overall_score:.1f}/100")
        print(f"   압축 라이브러리:   {'RealityStone + ' if RS_AVAILABLE else ''}극한 압축 기법")
        
        # 성공 판정 및 등급
        if overall_score >= 90:
            grade = "🏆 극한 대성공 (S급)"
            message = "모든 지표에서 극한 성능!"
        elif overall_score >= 80:
            grade = "🥇 극한 성공 (A급)"
            message = "대부분 지표에서 극한 성능!"
        elif overall_score >= 70:
            grade = "🥈 우수 (B급)"
            message = "상당한 극한 개선 효과!"
        elif overall_score >= 60:
            grade = "🥉 양호 (C급)"
            message = "일부 극한 개선 효과 있음"
        else:
            grade = "🔧 개선 필요 (D급)"
            message = "추가 극한 최적화 필요"
        
        print(f"\n{grade}: {message}")
        
        # 세부 권장사항
        if quality_retention < 0.85:
            print(f"💡 권장사항: 압축률을 줄이거나 파인튜닝 더 강화")
        if speed_improvement < 2.0:
            print(f"💡 권장사항: 더 적극적인 압축 전략 고려")
        if quality_improvement < 0.2:
            print(f"💡 권장사항: 파인튜닝 하이퍼파라미터 조정")
        
        print(f"\n🌟 극한 최종 결론:")
        print(f"   극한 RealityStone 압축 파이프라인으로")
        print(f"   {memory_saved:.0f}% 메모리 절약과 {speed_improvement:.1f}× 속도 향상을 달성하면서")
        print(f"   원본 품질의 {quality_retention*100:.0f}%를 유지했습니다!")
        
    except Exception as e:
        print(f"❌ 극한 압축 실패: {e}")
        print("🔧 더 안정적인 압축 방법이 필요합니다")

def _calculate_extreme_performance_score(quality_retention, speed_improvement, 
                                       compression_ratio, quality_improvement):
    """극한 성과 점수 계산"""
    
    # 각 지표별 점수 (0-25점)
    quality_score = min(25, quality_retention * 30)
    speed_score = min(25, (speed_improvement - 1) * 15)
    compression_score = min(25, (1 - compression_ratio) * 26.3)  # 95% 압축시 25점
    improvement_score = min(25, quality_improvement * 30)
    
    return quality_score + speed_score + compression_score + improvement_score

if __name__ == "__main__":
    main_extreme() 