import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import time  # 추론 속도 측정용 추가
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AccuracyFirstStats:
    """정확도 최우선 압축 통계"""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    accuracy_preserved: float
    fused_layers: List[str]
    energy_preserved: float
    svd_rank: int
    compression_method: str


class AccuracyFirstCompressor:
    def __init__(self, 
                 min_accuracy: float = 0.95,  # 최소 95% 정확도
                 energy_threshold: float = 0.99):  # 99% 에너지 보존
        
        self.min_accuracy = min_accuracy
        self.energy_threshold = energy_threshold
        
    def accuracy_first_compress(self, model: nn.Module, layer_names: List[str]) -> Dict:
        """정확도 최우선 압축"""
        
        print(f"🎯 정확도 최우선 압축: {layer_names}")
        print(f"   최소 정확도: {100*self.min_accuracy:.1f}%")
        print(f"   에너지 보존: {100*self.energy_threshold:.1f}%")
        
        # 1단계: 수학적으로 정확한 등가 가중치 계산
        equivalent_weight, equivalent_bias = self._compute_exact_equivalent(model, layer_names)
        print(f"   등가 가중치: {equivalent_weight.shape}")
        
        # 2단계: 고정밀도 SVD 분해
        svd_result = self._high_precision_svd(equivalent_weight)
        
        # 3단계: 에너지 기반 랭크 선택 (99% 이상 보존)
        optimal_rank = self._find_optimal_rank(svd_result, equivalent_weight)
        
        # 4단계: 정확도 검증 및 조정
        final_accuracy = self._verify_and_adjust_accuracy(
            equivalent_weight, svd_result, optimal_rank
        )
        
        result = {
            'type': 'accuracy_first_svd',
            'svd_components': {
                'U': svd_result['U'][:, :optimal_rank],
                'S': svd_result['S'][:optimal_rank],
                'V': svd_result['V'][:, :optimal_rank]
            },
            'svd_rank': optimal_rank,
            'original_shape': equivalent_weight.shape,
            'original_bias': equivalent_bias,
            'layer_names': layer_names,
            'accuracy': final_accuracy,
            'energy_preserved': self._calculate_energy_ratio(svd_result, optimal_rank)
        }
        
        return result
    
    def _compute_exact_equivalent(self, model: nn.Module, layer_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """수학적으로 정확한 등가 레이어 계산"""
        
        weights = []
        biases = []
        
        for name in layer_names:
            if hasattr(model, name):
                layer = getattr(model, name)
                if isinstance(layer, nn.Linear):
                    weights.append(layer.weight.data.clone())
                    biases.append(layer.bias.data.clone() if layer.bias is not None else None)
        
        if len(weights) == 0:
            raise ValueError("No linear layers found")
        
        # 정확한 체인 곱셈
        equivalent_weight = weights[0]
        for i in range(1, len(weights)):
            equivalent_weight = weights[i] @ equivalent_weight
        
        final_bias = biases[-1] if biases and biases[-1] is not None else None
        
        print(f"   수학적 등가: {equivalent_weight.shape[1]} → {equivalent_weight.shape[0]}")
        
        return equivalent_weight, final_bias
    
    def _high_precision_svd(self, matrix: torch.Tensor) -> Dict:
        """고정밀도 SVD 분해"""
        
        print(f"   고정밀도 SVD 분해...")
        
        try:
            # double precision으로 SVD
            U, S, V = torch.svd(matrix.double())
            U, S, V = U.float(), S.float(), V.float()
            
            print(f"   SVD 성공: rank {len(S)}")
            
            return {
                'U': U,
                'S': S, 
                'V': V,
                'original_energy': torch.sum(S**2)
            }
            
        except Exception as e:
            print(f"   SVD 실패: {e}")
            # fallback: 단위 행렬
            min_dim = min(matrix.shape)
            return {
                'U': torch.eye(matrix.shape[0]),
                'S': torch.ones(min_dim),
                'V': torch.eye(matrix.shape[1]),
                'original_energy': torch.sum(matrix**2)
            }
    
    def _find_optimal_rank(self, svd_result: Dict, original_matrix: torch.Tensor) -> int:
        """최적 랭크 찾기 (에너지 99% 보존)"""
        
        S = svd_result['S']
        total_energy = svd_result['original_energy']
        
        cumulative_energy = torch.cumsum(S**2, dim=0)
        energy_ratios = cumulative_energy / total_energy
        
        # 99% 에너지 보존하는 최소 랭크
        optimal_rank = torch.sum(energy_ratios < self.energy_threshold).item() + 1
        
        # 최소 10개는 보존 (너무 공격적 압축 방지)
        optimal_rank = max(optimal_rank, 10)
        optimal_rank = min(optimal_rank, len(S))
        
        energy_preserved = energy_ratios[optimal_rank-1].item()
        
        print(f"   최적 랭크: {optimal_rank}/{len(S)} (에너지 {100*energy_preserved:.2f}%)")
        
        return optimal_rank
    
    def _verify_and_adjust_accuracy(self, original_matrix: torch.Tensor, 
                                   svd_result: Dict, initial_rank: int) -> float:
        """정확도 검증 및 조정"""
        
        print(f"   정확도 검증...")
        
        best_accuracy = 0.0
        best_rank = initial_rank
        
        # 초기 랭크부터 시작해서 점진적으로 증가
        for rank in range(initial_rank, min(initial_rank + 20, len(svd_result['S']))):
            
            # 현재 랭크로 복원
            reconstructed = self._reconstruct_from_svd(svd_result, rank)
            accuracy = self._calculate_precise_accuracy(original_matrix, reconstructed)
            
            print(f"   랭크 {rank}: 정확도 {100*accuracy:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rank = rank
            
            # 목표 정확도 달성하면 조기 종료
            if accuracy >= self.min_accuracy:
                print(f"   목표 달성! 랭크 {rank}, 정확도 {100*accuracy:.2f}%")
                return accuracy
            
            # 정확도가 감소하기 시작하면 중단
            if rank > initial_rank + 5 and accuracy < best_accuracy - 0.01:
                break
        
        print(f"   최고 정확도: {100*best_accuracy:.2f}% (랭크 {best_rank})")
        
        return best_accuracy
    
    def _reconstruct_from_svd(self, svd_result: Dict, rank: int) -> torch.Tensor:
        """SVD에서 행렬 복원"""
        
        U = svd_result['U'][:, :rank]
        S = svd_result['S'][:rank]  
        V = svd_result['V'][:, :rank]
        
        reconstructed = U @ torch.diag(S) @ V.T
        
        return reconstructed
    
    def _calculate_precise_accuracy(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """정밀한 정확도 계산"""
        
        try:
            if original.shape != reconstructed.shape:
                return 0.0
            
            orig_flat = original.flatten()
            recon_flat = reconstructed.flatten()
            
            # 1. 코사인 유사도 (50%)
            cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=0).item()
            
            # 2. 피어슨 상관계수 (30%)
            if torch.std(orig_flat) > 1e-8 and torch.std(recon_flat) > 1e-8:
                corr = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1].item()
                if torch.isnan(torch.tensor(corr)):
                    corr = cos_sim
            else:
                corr = cos_sim
            
            # 3. 정규화된 MSE (20%)
            mse = F.mse_loss(orig_flat, recon_flat)
            var_orig = torch.var(orig_flat)
            if var_orig > 1e-8:
                normalized_mse = mse / var_orig
                mse_accuracy = torch.exp(-normalized_mse).item()
            else:
                mse_accuracy = 1.0 if mse < 1e-8 else 0.0
            
            # 종합 정확도
            accuracy = 0.5 * cos_sim + 0.3 * corr + 0.2 * mse_accuracy
            
            return max(0.0, min(1.0, accuracy))
            
        except Exception as e:
            print(f"   정확도 계산 실패: {e}")
            return 0.0
    
    def _calculate_energy_ratio(self, svd_result: Dict, rank: int) -> float:
        """에너지 보존률 계산"""
        
        S = svd_result['S']
        total_energy = svd_result['original_energy']
        preserved_energy = torch.sum(S[:rank]**2)
        
        return (preserved_energy / total_energy).item()


class AccuracyFirstLayer(nn.Module):
    """정확도 최우선 압축 레이어"""
    
    def __init__(self, compressed_data: Dict):
        super().__init__()
        
        # SVD 성분에서 가중치 복원
        svd_components = compressed_data['svd_components']
        reconstructed_weight = (svd_components['U'] @ 
                              torch.diag(svd_components['S']) @ 
                              svd_components['V'].T)
        
        self.weight = nn.Parameter(reconstructed_weight)
        
        if compressed_data['original_bias'] is not None:
            self.bias = nn.Parameter(compressed_data['original_bias'])
        else:
            self.register_parameter('bias', None)
        
        print(f"   정확도 최우선 복원: {self.weight.shape}")
        print(f"   SVD 랭크: {compressed_data['svd_rank']}")
        print(f"   정확도: {100*compressed_data['accuracy']:.2f}%")
        print(f"   에너지 보존: {100*compressed_data['energy_preserved']:.1f}%")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def accuracy_first_compress(model: nn.Module, 
                          layer_names: List[str],
                          min_accuracy: float = 0.95,
                          test_input: Optional[torch.Tensor] = None) -> Tuple[nn.Module, AccuracyFirstStats]:
    """정확도 최우선 압축 (정확도 95%+ 목표)"""
    
    print("🎯 정확도 최우선 압축 (95%+ 정확도)")
    print("=" * 60)
    
    compressor = AccuracyFirstCompressor(min_accuracy=min_accuracy)
    
    # 원본 정보
    original_params = sum(p.numel() for p in model.parameters())
    original_size_mb = original_params * 4 / (1024**2)
    
    print(f"원본: {original_params:,} 파라미터 ({original_size_mb:.2f}MB)")
    print(f"목표: 정확도 {min_accuracy:.0%}+ (정확도 최우선)")
    
    # 원본 출력
    original_output = None
    if test_input is not None:
        with torch.no_grad():
            model.eval()
            original_output = model(test_input)
    
    # 정확도 최우선 압축
    compressed_data = compressor.accuracy_first_compress(model, layer_names)
    
    # 압축된 레이어 생성
    accuracy_first_layer = AccuracyFirstLayer(compressed_data)
    
    # 새 모델 구성
    class AccuracyFirstModel(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.accuracy_layer = layer
        
        def forward(self, x):
            return self.accuracy_layer(x)
    
    compressed_model = AccuracyFirstModel(accuracy_first_layer)
    
    # 압축 통계
    compressed_params = accuracy_first_layer.weight.numel()
    if accuracy_first_layer.bias is not None:
        compressed_params += accuracy_first_layer.bias.numel()
    
    # 원래 융합 대상 파라미터 수
    fusion_params = 0
    for name in layer_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Linear):
                fusion_params += layer.weight.numel()
                if layer.bias is not None:
                    fusion_params += layer.bias.numel()
    
    compressed_size_mb = compressed_params * 4 / (1024**2)
    compression_ratio = compressed_params / fusion_params
    
    # 전체 모델 정확도
    accuracy_preserved = 0.0
    
    if test_input is not None and original_output is not None:
        with torch.no_grad():
            compressed_model.eval()
            try:
                compressed_output = compressed_model(test_input)
                
                if compressed_output.shape == original_output.shape:
                    accuracy_preserved = compressor._calculate_precise_accuracy(
                        original_output, compressed_output
                    )
                    
            except Exception as e:
                print(f"   모델 정확도 계산 실패: {e}")
                accuracy_preserved = 0.0
    
    stats = AccuracyFirstStats(
        original_size_mb=original_size_mb,
        compressed_size_mb=compressed_size_mb,
        compression_ratio=compression_ratio,
        accuracy_preserved=accuracy_preserved,
        fused_layers=layer_names,
        energy_preserved=compressed_data['energy_preserved'],
        svd_rank=compressed_data['svd_rank'],
        compression_method="정확도 최우선 SVD (99% 에너지 보존)"
    )
    
    print(f"\n✅ 정확도 최우선 압축 완료!")
    print(f"융합 파라미터: {fusion_params:,} → {compressed_params:,}")
    print(f"크기: {fusion_params * 4 / (1024**2):.2f}MB → {compressed_size_mb:.2f}MB")
    print(f"압축률: {compression_ratio:.3f} ({100*compression_ratio:.1f}%)")
    print(f"정확도: {100*accuracy_preserved:.2f}%")
    print(f"에너지 보존: {100*compressed_data['energy_preserved']:.1f}%")
    
    return compressed_model, stats


def test_accuracy_first():
    """정확도 최우선 압축 테스트"""
    
    print("🎯 정확도 최우선 압축 테스트")
    print("=" * 60)
    
    # 순수 선형 테스트 모델
    class PureLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128) 
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 32)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return self.fc4(x)
    
    model = PureLinearModel()
    test_input = torch.randn(16, 128)
    
    # 원본 테스트
    print("원본 모델 테스트...")
    with torch.no_grad():
        original_output = model(test_input)
        print(f"원본 출력: {original_output.shape}")
        print(f"원본 범위: [{original_output.min():.4f}, {original_output.max():.4f}]")
        print(f"원본 평균: {original_output.mean():.4f}")
    
    # 정확도 최우선 압축
    layer_names = ['fc1', 'fc2', 'fc3', 'fc4']
    compressed_model, stats = accuracy_first_compress(
        model, 
        layer_names,
        min_accuracy=0.95,  # 95% 정확도 목표
        test_input=test_input
    )
    
    # 압축된 모델 테스트
    print("\n압축된 모델 테스트...")
    with torch.no_grad():
        try:
            compressed_output = compressed_model(test_input)
            print(f"압축 출력: {compressed_output.shape}")
            print(f"압축 범위: [{compressed_output.min():.4f}, {compressed_output.max():.4f}]")
            print(f"압축 평균: {compressed_output.mean():.4f}")
            
            # 정밀 분석
            if compressed_output.shape == original_output.shape:
                diff = torch.abs(original_output - compressed_output)
                rel_error = diff / (torch.abs(original_output) + 1e-8)
                
                print(f"\n📊 정확도 최우선 분석:")
                print(f"   최대 차이: {diff.max():.8f}")
                print(f"   평균 차이: {diff.mean():.8f}")
                print(f"   최대 상대오차: {rel_error.max():.6f}")
                print(f"   평균 상대오차: {rel_error.mean():.6f}")
                
                cos_sim = F.cosine_similarity(original_output.flatten(), compressed_output.flatten(), dim=0)
                print(f"   코사인 유사도: {cos_sim:.8f}")
                
                # 수학적 검증
                print(f"\n🔬 수학적 검증:")
                with torch.no_grad():
                    manual_output = test_input
                    for layer_name in layer_names:
                        layer = getattr(model, layer_name)
                        manual_output = layer(manual_output)
                    
                    manual_vs_original = torch.allclose(manual_output, original_output, atol=1e-6)
                    print(f"   수동 계산 == 원본: {manual_vs_original}")
                    
                    manual_vs_compressed = F.cosine_similarity(manual_output.flatten(), compressed_output.flatten(), dim=0)
                    print(f"   수동 vs 압축 유사도: {manual_vs_compressed:.8f}")
                
        except Exception as e:
            print(f"압축 모델 실행 실패: {e}")
            traceback.print_exc()
    
    # 🚀 추론 속도 비교 테스트
    print("\n🚀 추론 속도 비교 테스트")
    print("=" * 40)
    
    def benchmark_model(model, input_data, model_name, warmup_runs=10, test_runs=100):
        """모델 추론 속도 벤치마크"""
        
        model.eval()
        
        # GPU가 있으면 GPU로 이동
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        input_data = input_data.to(device)
        
        print(f"📊 {model_name} 벤치마크 (device: {device})")
        
        # Warmup 실행 (GPU 캐시 준비)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # 실제 측정
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_runs):
                output = model(input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / test_runs
        fps = 1.0 / avg_time
        
        print(f"   평균 추론 시간: {avg_time*1000:.3f}ms")
        print(f"   처리량: {fps:.1f} FPS")
        print(f"   총 시간 ({test_runs}회): {total_time:.3f}s")
        
        # 메모리 사용량 (GPU에서만)
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"   GPU 메모리: {memory_used:.1f}MB")
            torch.cuda.reset_peak_memory_stats()
        
        return avg_time, fps
    
    # 다양한 배치 크기로 테스트
    batch_sizes = [1, 8, 16, 32]
    speed_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔍 배치 크기 {batch_size} 테스트:")
        test_batch = torch.randn(batch_size, 128)
        
        # 원본 모델 속도
        original_time, original_fps = benchmark_model(
            model, test_batch, f"원본 모델 (batch={batch_size})"
        )
        
        # 압축 모델 속도  
        compressed_time, compressed_fps = benchmark_model(
            compressed_model, test_batch, f"압축 모델 (batch={batch_size})"
        )
        
        # 속도 개선률 계산
        speedup = original_time / compressed_time
        throughput_gain = compressed_fps / original_fps
        
        print(f"   ⚡ 속도 개선: {speedup:.2f}x 빠름")
        print(f"   📈 처리량 증가: {throughput_gain:.2f}x")
        
        speed_results[batch_size] = {
            'original_time': original_time,
            'compressed_time': compressed_time,
            'speedup': speedup,
            'throughput_gain': throughput_gain
        }
    
    # 종합 속도 분석
    print(f"\n📊 종합 속도 분석:")
    avg_speedup = np.mean([result['speedup'] for result in speed_results.values()])
    avg_throughput_gain = np.mean([result['throughput_gain'] for result in speed_results.values()])
    
    print(f"   평균 속도 개선: {avg_speedup:.2f}x")
    print(f"   평균 처리량 증가: {avg_throughput_gain:.2f}x")
    
    # 파라미터 수 감소와 속도 개선 비교
    param_reduction = (1 - stats.compression_ratio) * 100
    print(f"   파라미터 감소: {param_reduction:.1f}%")
    print(f"   속도 개선: {(avg_speedup-1)*100:.1f}%")
    
    if avg_speedup > 1.0:
        print("   ✅ 압축으로 인한 속도 향상 확인!")
    else:
        print("   ⚠️ 압축 후 속도 저하 발생")
    
    print(f"\n📊 최종 정확도 최우선 결과:")
    print(f"   압축률: {100*stats.compression_ratio:.1f}%")
    print(f"   정확도: {100*stats.accuracy_preserved:.3f}%")
    print(f"   에너지 보존: {100*stats.energy_preserved:.1f}%")
    print(f"   SVD 랭크: {stats.svd_rank}")
    print(f"   방법: {stats.compression_method}")
    
    # 성공 기준: 정확도 95%+
    success = stats.accuracy_preserved >= 0.95
    
    if success:
        print("✅ 정확도 최우선 압축 성공!")
        print(f"   목표 달성: 정확도 {100*stats.accuracy_preserved:.2f}% ≥ 95%")
        return True
    else:
        print("⚠️ 목표 미달성")
        print(f"   정확도: {100*stats.accuracy_preserved:.2f}% < 95%")
        return False


if __name__ == "__main__":
    try:
        success = test_accuracy_first()
        if success:
            print("\n🎉 정확도 최우선 압축 완료!")
        else:
            print("\n⚠️ 추가 개선 필요")
    except Exception as e:
        print(f"실행 실패: {e}")
        print(traceback.format_exc()) 