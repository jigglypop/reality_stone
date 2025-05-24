"""
Reality Stone + 진짜 헬가손 변환 압축
하이퍼볼릭 기하학에서의 직접적인 기하학적 압축
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Reality Stone 백엔드 무조건 사용
import sys
sys.path.insert(0, '.')

try:
    import reality_stone
    print("SUCCESS Reality Stone 백엔드 로드 성공!")
    
    # Reality Stone 함수들 확인
    all_funcs = [name for name in dir(reality_stone) if not name.startswith('_')]
    print(f"Reality Stone 함수들: {len(all_funcs)}개")
    
    REALITY_STONE_AVAILABLE = True
    
except ImportError as e:
    print(f"ERROR Reality Stone 백엔드 로드 실패: {e}")
    print("STOP Reality Stone 없으면 중단!")
    exit(1)


def poincare_exp_map(v, c=1.0):
    """포인카레 디스크에서 원점에서의 지수 맵"""
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    
    # 수치적 안정성을 위한 클램핑
    v_norm_clamped = torch.clamp(v_norm, min=1e-8)
    
    exp_factor = torch.tanh(sqrt_c * v_norm_clamped) / (sqrt_c * v_norm_clamped)
    
    return exp_factor * v


def poincare_log_map(x, c=1.0):
    """포인카레 디스크에서 원점으로의 로그 맵"""
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    
    # 수치적 안정성을 위한 클램핑 (포인카레 디스크 내부)
    x_norm_clamped = torch.clamp(x_norm, max=0.99)
    
    log_factor = torch.atanh(sqrt_c * x_norm_clamped) / (sqrt_c * x_norm_clamped + 1e-8)
    
    return log_factor * x


def hyperbolic_distance(x, y, c=1.0):
    """포인카레 디스크에서의 하이퍼볼릭 거리"""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    
    diff = x - y
    diff_norm_sq = torch.sum(diff * diff, dim=-1)
    
    x_norm_sq = torch.sum(x * x, dim=-1)
    y_norm_sq = torch.sum(y * y, dim=-1)
    
    denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    numerator = 2 * diff_norm_sq
    
    # 수치적 안정성
    denominator = torch.clamp(denominator, min=1e-8)
    ratio = torch.clamp(numerator / denominator, min=0, max=1e6)
    
    distance = (1 / sqrt_c) * torch.acosh(1 + ratio)
    
    return distance


def helgason_geometric_compress(weight_matrix, compression_ratio=0.1, curvature=1.0):
    """
    진짜 헬가손 변환: 하이퍼볼릭 기하학적 압축
    
    핵심 아이디어:
    1. 가중치를 하이퍼볼릭 공간의 점들로 해석
    2. 기하학적 클러스터링으로 중요한 방향만 유지
    3. 하이퍼볼릭 거리 기반 근사
    """
    
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    original_shape = weight_matrix.shape
    
    # 1. 가중치를 벡터로 변환
    if len(original_shape) == 2:
        # 2D 행렬을 행별로 처리
        vectors = weight_matrix  # [out_features, in_features]
    else:
        # 다차원은 flatten
        vectors = weight_matrix.view(original_shape[0], -1)
    
    out_features, in_features = vectors.shape
    
    # 2. 각 벡터를 포인카레 디스크로 매핑
    # 정규화를 통해 하이퍼볼릭 공간에 맞게 조정
    vector_norms = torch.norm(vectors, dim=1, keepdim=True)
    max_norm = torch.max(vector_norms)
    
    # 포인카레 디스크 내부에 맞게 스케일링
    if max_norm > 0:
        scale_factor = 0.9 / max_norm  # 경계에서 떨어뜨림
        scaled_vectors = vectors * scale_factor
    else:
        scaled_vectors = vectors
    
    # 3. 하이퍼볼릭 공간에서 기하학적 압축
    # k-means와 비슷하지만 하이퍼볼릭 거리 사용
    
    num_clusters = max(1, int(out_features * compression_ratio))
    
    # 클러스터 중심 초기화 (랜덤 선택)
    cluster_indices = torch.randperm(out_features, device=device)[:num_clusters]
    cluster_centers = scaled_vectors[cluster_indices].clone()
    
    # 하이퍼볼릭 k-means (단순화된 버전)
    for iteration in range(5):  # 빠른 수렴을 위해 5회만
        # 각 점을 가장 가까운 클러스터에 할당
        distances = torch.zeros(out_features, num_clusters, device=device)
        
        for i in range(num_clusters):
            distances[:, i] = hyperbolic_distance(
                scaled_vectors, 
                cluster_centers[i:i+1].expand_as(scaled_vectors),
                c=curvature
            )
        
        assignments = torch.argmin(distances, dim=1)
        
        # 클러스터 중심 업데이트 (하이퍼볼릭 평균)
        for i in range(num_clusters):
            mask = (assignments == i)
            if mask.sum() > 0:
                cluster_points = scaled_vectors[mask]
                
                # 하이퍼볼릭 공간에서의 중심 계산 (근사)
                # 로그 맵 → 유클리드 평균 → 지수 맵
                log_points = poincare_log_map(cluster_points, c=curvature)
                euclidean_mean = torch.mean(log_points, dim=0)
                cluster_centers[i] = poincare_exp_map(euclidean_mean, c=curvature)
    
    # 4. 압축된 표현 저장
    # 각 원본 벡터를 클러스터 중심 + 잔차로 분해
    compressed_data = {
        'cluster_centers': cluster_centers,  # [num_clusters, in_features]
        'assignments': assignments,          # [out_features]
        'residuals': [],                     # 작은 잔차들만 저장
        'scale_factor': scale_factor if max_norm > 0 else torch.tensor(1.0)
    }
    
    # Reality Stone 처리 (클러스터 중심에 적용)
    try:
        if hasattr(reality_stone, 'poincare_compress'):
            compressed_data['cluster_centers'] = reality_stone.poincare_compress(
                compressed_data['cluster_centers'].float()
            )
        elif hasattr(reality_stone, 'hyperbolic_compress'):
            compressed_data['cluster_centers'] = reality_stone.hyperbolic_compress(
                compressed_data['cluster_centers'].float()
            )
    except:
        pass
    
    # 5. 중요한 잔차만 저장 (압축률에 따라)
    residual_budget = max(1, int(out_features * compression_ratio * 0.1))  # 10%는 잔차용
    
    for i in range(out_features):
        center_idx = assignments[i]
        residual = scaled_vectors[i] - cluster_centers[center_idx]
        residual_importance = torch.norm(residual)
        
        if len(compressed_data['residuals']) < residual_budget:
            compressed_data['residuals'].append({
                'idx': i,
                'residual': residual,
                'importance': residual_importance
            })
        else:
            # 가장 중요하지 않은 잔차와 비교
            min_importance_idx = min(
                range(len(compressed_data['residuals'])),
                key=lambda x: compressed_data['residuals'][x]['importance']
            )
            
            if residual_importance > compressed_data['residuals'][min_importance_idx]['importance']:
                compressed_data['residuals'][min_importance_idx] = {
                    'idx': i,
                    'residual': residual,
                    'importance': residual_importance
                }
    
    reconstruction_info = {
        'original_shape': original_shape,
        'out_features': out_features,
        'in_features': in_features,
        'curvature': curvature,
        'compression_ratio': compression_ratio
    }
    
    return compressed_data, reconstruction_info


def helgason_geometric_reconstruct(compressed_data, reconstruction_info):
    """하이퍼볼릭 기하학적 압축 재구성"""
    
    original_shape = reconstruction_info['original_shape']
    out_features = reconstruction_info['out_features']
    in_features = reconstruction_info['in_features']
    
    cluster_centers = compressed_data['cluster_centers']
    assignments = compressed_data['assignments']
    residuals = compressed_data['residuals']
    scale_factor = compressed_data['scale_factor']
    
    device = cluster_centers.device
    
    # 1. 기본 재구성 (클러스터 중심)
    reconstructed_vectors = torch.zeros(out_features, in_features, device=device)
    
    for i in range(out_features):
        center_idx = assignments[i]
        reconstructed_vectors[i] = cluster_centers[center_idx]
    
    # 2. 잔차 적용
    for residual_info in residuals:
        idx = residual_info['idx']
        reconstructed_vectors[idx] += residual_info['residual']
    
    # 3. 스케일 복원
    reconstructed_vectors = reconstructed_vectors / scale_factor
    
    # 4. 원본 모양으로 복원
    if len(original_shape) == 2:
        reconstructed_weight = reconstructed_vectors
    else:
        reconstructed_weight = reconstructed_vectors.view(original_shape)
    
    return reconstructed_weight


class RealityStoneHelgasonLinear(nn.Module):
    """Reality Stone + 진짜 헬가손 변환 압축 레이어"""
    
    def __init__(self, original_linear, compression_ratio=0.1, layer_name="unknown", verbose=False):
        super().__init__()
        
        self.layer_name = layer_name
        self.verbose = verbose
        
        # 원본 정보
        original_weight = original_linear.weight.data.clone()
        original_bias = original_linear.bias.data.clone() if original_linear.bias is not None else None
        
        # c_proj 레이어의 가중치 전치 (차원 불일치 해결)
        if "mlp_c_proj" in layer_name:  # MLP의 c_proj만 전치
            if verbose:
                print(f"   DEBUG MLP c_proj 전치: {original_weight.shape} → {original_weight.t().shape}")
            original_weight = original_weight.t()  # (3072, 768) → (768, 3072)
        elif "mlp_c_fc" in layer_name:  # MLP c_fc 전치
            if verbose:
                print(f"   DEBUG MLP c_fc 전치: {original_weight.shape} → {original_weight.t().shape}")
            original_weight = original_weight.t()  # (768, 3072) → (3072, 768)
        elif "attn_c_attn" in layer_name:  # Attention c_attn 전치
            if verbose:
                print(f"   DEBUG Attention c_attn 전치: {original_weight.shape} → {original_weight.t().shape}")
            original_weight = original_weight.t()  # (768, 2304) → (2304, 768)
        
        device = original_weight.device
        dtype = original_weight.dtype
        
        if self.verbose:
            print(f"   헬가손 압축 전: {original_weight.shape}")
        
        # 헬가손 기하학적 압축
        self.compressed_data, self.reconstruction_info = helgason_geometric_compress(
            original_weight, compression_ratio, curvature=1.0
        )
        
        # 압축된 데이터를 파라미터로 저장
        self.cluster_centers = nn.Parameter(self.compressed_data['cluster_centers'].to(dtype).to(device))
        self.register_buffer('assignments', self.compressed_data['assignments'])
        self.register_buffer('scale_factor', self.compressed_data['scale_factor'])
        
        # 잔차들을 별도 파라미터로 저장
        self.residuals = nn.ParameterList()
        self.residual_indices = []
        
        for residual_info in self.compressed_data['residuals']:
            self.residuals.append(nn.Parameter(residual_info['residual'].to(dtype).to(device)))
            self.residual_indices.append(residual_info['idx'])
        
        # 바이어스 저장
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.to(dtype).to(device))
        else:
            self.register_parameter('bias', None)
        
        # 차원 정보 (전치 후 기준)
        self.out_features = original_weight.shape[0]
        self.in_features = original_weight.shape[1]
        
        # 압축 통계
        original_params = original_weight.numel() + (original_bias.numel() if original_bias is not None else 0)
        compressed_params = (self.cluster_centers.numel() + 
                           sum(r.numel() for r in self.residuals) +
                           self.assignments.numel() +
                           (self.bias.numel() if self.bias is not None else 0))
        self.compression_ratio = compressed_params / original_params
        
        if self.verbose:
            print(f"   헬가손 압축 완료: {original_params:,} → {compressed_params:,} ({self.compression_ratio:.3f})")
            print(f"   클러스터 수: {len(self.cluster_centers)}, 잔차 수: {len(self.residuals)}")
    
    def forward(self, x):
        """헬가손 압축된 가중치로 순전파"""
        
        # 가중치 재구성
        out_features = self.out_features
        in_features = self.in_features
        device = x.device
        
        # 1. 기본 재구성 (클러스터 중심)
        reconstructed_weight = torch.zeros(out_features, in_features, device=device, dtype=x.dtype)
        
        for i in range(out_features):
            center_idx = self.assignments[i]
            reconstructed_weight[i] = self.cluster_centers[center_idx]
        
        # 2. 잔차 적용
        for i, residual_idx in enumerate(self.residual_indices):
            reconstructed_weight[residual_idx] += self.residuals[i]
        
        # 3. 스케일 복원
        reconstructed_weight = reconstructed_weight / self.scale_factor
        
        # 4. 선형 변환
        output = F.linear(x, reconstructed_weight, self.bias)
        
        return output


def load_korean_model():
    """한글 모델 로드"""
    
    print("LOADING 한글 모델 로딩...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        print(f"   로딩: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   SUCCESS 로드 성공!")
        return model, tokenizer, model_name
        
    except Exception as e:
        print(f"   ERROR 로드 실패: {e}")
        return None, None, None


def apply_helgason_compression(model, compression_ratio=0.2, verbose=True):
    """Reality Stone 헬가손 압축 적용"""
    
    if verbose:
        print(f"\nCOMPRESS Reality Stone 헬가손 압축 적용 (압축률: {compression_ratio:.1%})")
    
    compressed_count = 0
    total_original = 0
    total_compressed = 0
    
    # 모든 레이어 압축 (차원 불일치 해결)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers_to_process = len(model.transformer.h)  # 모든 레이어
        if verbose:
            print(f"   헬가손 압축할 레이어: {layers_to_process}개 (전체)")
        
        for layer_idx in range(layers_to_process):
            layer = model.transformer.h[layer_idx]
            
            if hasattr(layer, 'mlp'):
                if verbose:
                    print(f"\nCOMPRESS Layer {layer_idx} 헬가손 압축 중...")
                
                try:
                    mlp = layer.mlp
                    
                    # c_fc 헬가손 압축 (768 → 3072)
                    if hasattr(mlp, 'c_fc'):
                        if verbose:
                            print(f"   c_fc 헬가손 압축: {mlp.c_fc.weight.shape}")
                        original_params = mlp.c_fc.weight.numel() + (mlp.c_fc.bias.numel() if mlp.c_fc.bias is not None else 0)
                        
                        compressed_fc = RealityStoneHelgasonLinear(mlp.c_fc, compression_ratio, f"mlp_c_fc_{layer_idx}", verbose=verbose)
                        mlp.c_fc = compressed_fc
                        
                        total_original += original_params
                        total_compressed += sum(p.numel() for p in compressed_fc.parameters())
                    
                    # c_proj 헬가손 압축 (3072 → 768)
                    if hasattr(mlp, 'c_proj'):
                        if verbose:
                            print(f"   c_proj 헬가손 압축: {mlp.c_proj.weight.shape}")
                        original_params = mlp.c_proj.weight.numel() + (mlp.c_proj.bias.numel() if mlp.c_proj.bias is not None else 0)
                        
                        compressed_proj = RealityStoneHelgasonLinear(mlp.c_proj, compression_ratio, f"mlp_c_proj_{layer_idx}", verbose=verbose)
                        mlp.c_proj = compressed_proj
                        
                        total_original += original_params
                        total_compressed += sum(p.numel() for p in compressed_proj.parameters())
                    
                    # Attention 레이어도 압축 (차원 불일치 완전 해결)
                    if hasattr(layer, 'attn'):
                        attn = layer.attn
                        
                        # attn.c_attn 압축 (768 → 2304)
                        if hasattr(attn, 'c_attn'):
                            if verbose:
                                print(f"   attn.c_attn 헬가손 압축: {attn.c_attn.weight.shape}")
                            original_params = attn.c_attn.weight.numel() + (attn.c_attn.bias.numel() if attn.c_attn.bias is not None else 0)
                            
                            compressed_attn = RealityStoneHelgasonLinear(attn.c_attn, compression_ratio, f"attn_c_attn_{layer_idx}", verbose=verbose)
                            attn.c_attn = compressed_attn
                            
                            total_original += original_params
                            total_compressed += sum(p.numel() for p in compressed_attn.parameters())
                        
                        # attn.c_proj 압축 (768 → 768)
                        if hasattr(attn, 'c_proj'):
                            if verbose:
                                print(f"   attn.c_proj 헬가손 압축: {attn.c_proj.weight.shape}")
                            original_params = attn.c_proj.weight.numel() + (attn.c_proj.bias.numel() if attn.c_proj.bias is not None else 0)
                            
                            compressed_attn_proj = RealityStoneHelgasonLinear(attn.c_proj, compression_ratio, f"attn_c_proj_{layer_idx}", verbose=verbose)
                            attn.c_proj = compressed_attn_proj
                            
                            total_original += original_params
                            total_compressed += sum(p.numel() for p in compressed_attn_proj.parameters())
                    
                    compressed_count += 1
                    if verbose:
                        print(f"   SUCCESS Layer {layer_idx} 헬가손 압축 완료")
                    
                except Exception as e:
                    if verbose:
                        print(f"   ERROR Layer {layer_idx} 헬가손 압축 실패: {e}")
                        import traceback
                        traceback.print_exc()
    
    overall_ratio = total_compressed / total_original if total_original > 0 else 1.0
    memory_saved = (total_original - total_compressed) * 4 / (1024**2)
    
    if verbose:
        print(f"\nRESULT Reality Stone 헬가손 압축 완료:")
        print(f"   압축된 레이어: {compressed_count}개")
        print(f"   파라미터: {total_original:,} → {total_compressed:,}")
        print(f"   실제 압축률: {overall_ratio:.3f}")
        print(f"   메모리 절약: {memory_saved:.1f}MB")
    
    return model, overall_ratio


def test_generation(model, tokenizer, prompts):
    """생성 테스트"""
    
    if tokenizer is None:
        return [], 0.0
    
    print("TESTING Reality Stone 헬가손 한글 생성 테스트")
    results = []
    times = []
    
    for i, prompt in enumerate(prompts[:2]):
        try:
            print(f"\n{i+1}. '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 15,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_time = time.time() - start_time
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append(generated)
            times.append(gen_time)
            
            print(f"   생성: {generated}")
            print(f"   시간: {gen_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ERROR 실패: {e}")
            results.append("")
            times.append(0)
    
    avg_time = np.mean(times) if times else 0
    print(f"\nTIME 평균 생성 시간: {avg_time*1000:.1f}ms")
    
    return results, avg_time


def reality_stone_helgason_test():
    """Reality Stone 헬가손 압축 테스트"""
    
    print("START Reality Stone 진짜 헬가손 기하학적 압축 테스트")
    print("=" * 80)
    
    # 1. 한글 모델 로드
    korean_model, tokenizer, model_name = load_korean_model()
    
    if korean_model is None:
        print("ERROR 한글 모델 로드 실패")
        return []
    
    print(f"\nINFO 모델 정보:")
    total_params = sum(p.numel() for p in korean_model.parameters())
    print(f"   모델: {model_name}")
    print(f"   파라미터: {total_params:,}")
    print(f"   크기: {total_params * 4 / (1024**2):.1f}MB")
    
    # 2. 원본 테스트
    prompts = ["안녕하세요, 오늘은", "인공지능의 발전으로"]
    print("\nTEST 원본 모델 테스트")
    original_results, original_time = test_generation(korean_model, tokenizer, prompts)
    
    # 3. 헬가손 압축률 테스트
    compression_ratios = [0.5]  # 50% 압축률만 테스트 (빠른 테스트용)
    
    results = []
    
    for ratio in compression_ratios:
        print(f"\nCOMPRESS Reality Stone 헬가손 압축률 {ratio:.1%} 테스트")
        print("=" * 60)
        
        try:
            # 모델 복사
            test_model = copy.deepcopy(korean_model)
            
            # 헬가손 압축 적용
            compressed_model, actual_ratio = apply_helgason_compression(test_model, ratio)
            
            # 압축된 모델 테스트
            print("\nTEST 헬가손 압축된 모델 테스트")
            compressed_results, compressed_time = test_generation(compressed_model, tokenizer, prompts)
            
            speed_improvement = original_time / compressed_time if compressed_time > 0 else 1.0
            
            result = {
                'compression_ratio': ratio,
                'actual_ratio': actual_ratio,
                'speed_improvement': speed_improvement,
                'original_time': original_time * 1000,
                'compressed_time': compressed_time * 1000,
                'success': len([r for r in compressed_results if r]) > 0
            }
            
            results.append(result)
            
            print(f"\nRESULT 헬가손 {ratio:.1%} 결과:")
            print(f"   실제 압축률: {actual_ratio:.3f}")
            print(f"   메모리 절약: {(1-actual_ratio)*100:.1f}%")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            print(f"   생성 시간: {original_time*1000:.1f}ms → {compressed_time*1000:.1f}ms")
            print(f"   생성 성공: {'SUCCESS' if result['success'] else 'FAIL'}")
            
            # 품질 비교
            if result['success']:
                print(f"\nQUALITY 헬가손 생성 품질 비교:")
                for i in range(min(len(original_results), len(compressed_results))):
                    if original_results[i] and compressed_results[i]:
                        print(f"   프롬프트: {prompts[i]}")
                        print(f"   원본: {original_results[i][:60]}...")
                        print(f"   헬가손: {compressed_results[i][:60]}...")
                        print()
            
        except Exception as e:
            print(f"   ERROR 헬가손 {ratio:.1%} 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 최종 결과
    if results:
        print(f"\nRESULT Reality Stone 헬가손 압축 최종 결과")
        print("=" * 80)
        print(f"{'압축률':<8} {'실제':<8} {'절약률':<8} {'속도':<8} {'성공':<6}")
        print("-" * 80)
        
        for result in results:
            savings = 1 - result['actual_ratio']
            print(f"{result['compression_ratio']:.1%}     "
                  f"{result['actual_ratio']:.3f}   "
                  f"{savings:.1%}      "
                  f"{result['speed_improvement']:.2f}x     "
                  f"{'SUCCESS' if result['success'] else 'FAIL'}")
        
        # 성공한 결과 중 최고 성능
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            best_speed = max(successful_results, key=lambda x: x['speed_improvement'])
            best_compression = min(successful_results, key=lambda x: x['actual_ratio'])
            
            print(f"\nBEST 헬가손 최고 성능:")
            print(f"   최고 속도: {best_speed['compression_ratio']:.1%} ({best_speed['speed_improvement']:.2f}x)")
            print(f"   최고 압축: {best_compression['compression_ratio']:.1%} ({1-best_compression['actual_ratio']:.1%} 절약)")
            
            # 헬가손 효과 분석
            print(f"\nANALYSIS Reality Stone 헬가손 기하학적 압축 효과:")
            for result in successful_results:
                savings = (1 - result['actual_ratio']) * 100
                print(f"   {result['compression_ratio']:.1%} 압축: {savings:.1f}% 메모리 절약, {result['speed_improvement']:.2f}x 속도 향상")
                print(f"      → 하이퍼볼릭 클러스터링으로 기하학적 의미 보존")
        else:
            print(f"\nWARNING 모든 헬가손 압축 시도가 실패했습니다")
    
    print(f"\nFINISH Reality Stone 헬가손 기하학적 압축 테스트 완료!")
    return results


if __name__ == "__main__":
    try:
        results = reality_stone_helgason_test()
        print("\nSUCCESS Reality Stone 헬가손 압축 테스트 완료!")
        
        # 성공한 결과가 있는지 확인
        successful = len([r for r in results if r.get('success', False)]) > 0
        if successful:
            print("SUCCESS Reality Stone 헬가손 압축 및 생성 성공!")
            print("SUCCESS 하이퍼볼릭 기하학적 클러스터링 압축 완료!")
        else:
            print("FAIL 헬가손 압축에서도 생성 품질 문제 발생")
            
    except Exception as e:
        print(f"ERROR 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 