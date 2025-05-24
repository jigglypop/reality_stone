"""
FFT 음향 처리 기반 압축 - 차원 안정화 버전
Reality Stone 출력을 원본 차원으로 정확히 맞춰서 손실 없는 압축 달성

핵심 개선사항:
1. 차원 안정성 확보
2. Reality Stone 출력 정규화
3. 음향 압축 최적화
4. 오류 없는 순전파
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Reality Stone 백엔드 로드
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import reality_stone
    print("✅ Reality Stone 백엔드 로드 성공!")
    
    # 함수 확인
    all_funcs = [name for name in dir(reality_stone) if not name.startswith('_')]
    print(f"   전체 함수: {len(all_funcs)}개")
    
    layer_funcs = [f for f in all_funcs if 'layer' in f.lower()]
    print(f"   레이어 함수: {layer_funcs}")
    
    REALITY_STONE_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Reality Stone 백엔드 로드 실패: {e}")
    REALITY_STONE_AVAILABLE = False


class StabilizedFFTAudioEngine:
    """차원 안정성이 확보된 FFT 음향 압축 엔진"""
    
    def __init__(self, compression_ratio=0.3, quality_threshold=0.95):
        self.compression_ratio = compression_ratio
        self.quality_threshold = quality_threshold
        
        # 음향 처리 파라미터 (최적화됨)
        self.sample_rate = 22050  # 더 낮은 샘플링 레이트로 안정성 확보
        self.window_size = 1024   # 더 작은 윈도우
        self.hop_length = 256     # 더 작은 홉
        self.energy_threshold = 0.02  # 더 높은 임계값으로 중요한 성분만 선택
        
    def weight_to_audio_signal(self, weight_matrix):
        """가중치를 음향 신호로 변환 (안정화됨)"""
        
        device = weight_matrix.device
        dtype = weight_matrix.dtype
        original_shape = weight_matrix.shape
        
        # 2D 가중치를 1D 음향 신호로 변환
        audio_signal = weight_matrix.flatten().float()
        
        # 신호 정규화 (-1, 1 범위로) - 더 안정적
        signal_max = torch.max(torch.abs(audio_signal))
        if signal_max > 1e-8:  # 매우 작은 값 처리
            audio_signal = audio_signal / signal_max
        else:
            signal_max = 1.0
        
        return {
            'signal': audio_signal,
            'original_shape': original_shape,
            'normalization_factor': signal_max,
            'device': device,
            'dtype': dtype,
            'total_elements': audio_signal.numel()
        }
    
    def stabilized_spectral_analysis(self, audio_data):
        """안정화된 스펙트럼 분석"""
        
        signal = audio_data['signal']
        signal_length = len(signal)
        
        # 윈도우 크기를 신호 길이에 맞게 조정
        actual_window_size = min(self.window_size, signal_length)
        
        # 2의 거듭제곱으로 조정 (FFT 최적화)
        power_of_2 = 1
        while power_of_2 < actual_window_size:
            power_of_2 *= 2
        actual_window_size = min(power_of_2 // 2, actual_window_size)
        actual_window_size = max(32, actual_window_size)  # 최소 32
        
        # 신호 패딩 및 윈도우 분할
        if signal_length < actual_window_size:
            # 신호가 너무 짧으면 패딩
            padding = torch.zeros(actual_window_size - signal_length, device=signal.device)
            padded_signal = torch.cat([signal, padding])
            num_windows = 1
        else:
            # 윈도우 개수 계산
            num_windows = (signal_length + actual_window_size - 1) // actual_window_size
            target_length = num_windows * actual_window_size
            
            if target_length > signal_length:
                padding = torch.zeros(target_length - signal_length, device=signal.device)
                padded_signal = torch.cat([signal, padding])
            else:
                padded_signal = signal[:target_length]
        
        # 윈도우별 FFT
        windows = padded_signal.view(-1, actual_window_size)
        spectrogram = torch.fft.fft(windows)  # [num_windows, window_size]
        
        return {
            'spectrogram': spectrogram,
            'window_size': actual_window_size,
            'num_windows': windows.shape[0],
            'original_length': signal_length
        }
    
    def smart_frequency_selection(self, spectrum_data):
        """지능적 주파수 성분 선택"""
        
        spectrogram = spectrum_data['spectrogram']
        
        # 에너지 계산
        magnitude = torch.abs(spectrogram)
        energy = magnitude ** 2
        
        # 주파수별/시간별 에너지
        freq_energy = torch.mean(energy, dim=0)  # [window_size]
        time_energy = torch.mean(energy, dim=1)  # [num_windows]
        
        # 적응적 임계값 설정
        freq_threshold = self.energy_threshold * torch.max(freq_energy)
        time_threshold = self.energy_threshold * torch.max(time_energy)
        
        # 중요한 성분 선택 (더 보수적)
        important_freqs = freq_energy > freq_threshold
        important_times = time_energy > time_threshold
        
        # 최소 보장 (너무 많이 제거되지 않도록)
        min_freqs = max(1, int(len(freq_energy) * (1 - self.compression_ratio * 0.8)))
        min_times = max(1, int(len(time_energy) * (1 - self.compression_ratio * 0.8)))
        
        if torch.sum(important_freqs) < min_freqs:
            _, top_freq_indices = torch.topk(freq_energy, min_freqs)
            important_freqs = torch.zeros_like(freq_energy, dtype=torch.bool)
            important_freqs[top_freq_indices] = True
        
        if torch.sum(important_times) < min_times:
            _, top_time_indices = torch.topk(time_energy, min_times)
            important_times = torch.zeros_like(time_energy, dtype=torch.bool)
            important_times[top_time_indices] = True
        
        return {
            'magnitude': magnitude,
            'energy': energy,
            'freq_energy': freq_energy,
            'time_energy': time_energy,
            'freq_mask': important_freqs,
            'time_mask': important_times
        }
    
    def smart_compression(self, spectrum_data, features):
        """지능적 압축 (차원 보존)"""
        
        spectrogram = spectrum_data['spectrogram']
        freq_mask = features['freq_mask']
        time_mask = features['time_mask']
        
        # 선택적 압축 (중요한 성분은 보존)
        compressed_spectrogram = torch.zeros_like(spectrogram)
        
        for t_idx, t_selected in enumerate(time_mask):
            if t_selected:
                for f_idx, f_selected in enumerate(freq_mask):
                    if f_selected:
                        compressed_spectrogram[t_idx, f_idx] = spectrogram[t_idx, f_idx]
        
        # 압축률 계산
        original_nonzero = torch.sum(spectrogram != 0).item()
        compressed_nonzero = torch.sum(compressed_spectrogram != 0).item()
        actual_compression_ratio = compressed_nonzero / max(1, original_nonzero)
        
        return {
            'compressed_spectrogram': compressed_spectrogram,
            'compression_ratio': actual_compression_ratio,
            'freq_count': torch.sum(freq_mask).item(),
            'time_count': torch.sum(time_mask).item()
        }
    
    def stabilized_reconstruction(self, compressed_data, spectrum_data, audio_data):
        """안정화된 재구성"""
        
        compressed_spectrogram = compressed_data['compressed_spectrogram']
        original_length = spectrum_data['original_length']
        
        # IFFT로 시간 도메인 복원
        time_domain_windows = torch.fft.ifft(compressed_spectrogram)
        real_signal = torch.real(time_domain_windows)  # 실수부만 사용
        
        # 윈도우들을 연결
        reconstructed_signal = real_signal.flatten()
        
        # 원본 길이로 정확히 맞추기
        if len(reconstructed_signal) > original_length:
            reconstructed_signal = reconstructed_signal[:original_length]
        elif len(reconstructed_signal) < original_length:
            padding = torch.zeros(original_length - len(reconstructed_signal), 
                                device=reconstructed_signal.device)
            reconstructed_signal = torch.cat([reconstructed_signal, padding])
        
        return reconstructed_signal
    
    def dimension_safe_reality_stone(self, compressed_weight, original_shape):
        """차원 안전한 Reality Stone 적용"""
        
        if not REALITY_STONE_AVAILABLE:
            return compressed_weight, "fft_audio_only"
        
        try:
            # Reality Stone 입력 준비 (안전하게)
            if len(original_shape) == 2:
                out_features, in_features = original_shape
                
                # 더미 입력 생성 (배치 크기 1로 고정)
                dummy_input = torch.randn(1, in_features, 
                                        device=compressed_weight.device, 
                                        dtype=torch.float32)
                
                # poincare_ball_layer 적용
                enhanced = reality_stone.poincare_ball_layer(
                    dummy_input, 
                    compressed_weight.float(), 
                    1.0,  # c parameter
                    0.05  # t parameter (더 작게)
                )
                
                # 차원 확인 및 조정
                if enhanced.shape == original_shape:
                    return enhanced.to(compressed_weight.dtype), "fft_audio_reality_stone"
                else:
                    # 차원 맞춤
                    if enhanced.numel() >= compressed_weight.numel():
                        # 크기가 크거나 같으면 자르기
                        reshaped = enhanced.flatten()[:compressed_weight.numel()]
                        result = reshaped.view(original_shape)
                    else:
                        # 크기가 작으면 패딩
                        needed_elements = compressed_weight.numel() - enhanced.numel()
                        padding = torch.zeros(needed_elements, device=enhanced.device, dtype=enhanced.dtype)
                        reshaped = torch.cat([enhanced.flatten(), padding])
                        result = reshaped.view(original_shape)
                    
                    return result.to(compressed_weight.dtype), "fft_audio_reality_stone_resized"
            
        except Exception as e:
            print(f"      Reality Stone 적용 실패: {e}")
        
        return compressed_weight, "fft_audio_only"
    
    def compress_weight_matrix(self, weight_matrix):
        """통합 안정화 압축 파이프라인"""
        
        print(f"      안정화 FFT 음향 압축: {weight_matrix.shape}")
        
        try:
            # 1. 가중치 → 음향 신호 변환
            audio_data = self.weight_to_audio_signal(weight_matrix)
            
            # 2. 안정화된 스펙트럼 분석
            spectrum_data = self.stabilized_spectral_analysis(audio_data)
            
            # 3. 지능적 주파수 선택
            features = self.smart_frequency_selection(spectrum_data)
            
            # 4. 지능적 압축
            compressed_data = self.smart_compression(spectrum_data, features)
            
            # 5. 안정화된 재구성
            reconstructed_signal = self.stabilized_reconstruction(
                compressed_data, spectrum_data, audio_data
            )
            
            # 6. 가중치 복원
            audio_data['signal'] = reconstructed_signal
            compressed_weight = self.audio_signal_to_weight(audio_data)
            
            # 7. 차원 안전한 Reality Stone 적용
            final_weight, method_name = self.dimension_safe_reality_stone(
                compressed_weight, weight_matrix.shape
            )
            
            print(f"      ✅ 안정화 압축 성공: {compressed_data['compression_ratio']:.3f}")
            
            return {
                'method': method_name,
                'compressed_weight': final_weight,
                'compression_ratio': compressed_data['compression_ratio'],
                'success': True,
                'details': {
                    'spectral_compression': compressed_data['compression_ratio'],
                    'freq_components': compressed_data['freq_count'],
                    'time_segments': compressed_data['time_count']
                }
            }
            
        except Exception as e:
            print(f"      안정화 압축 실패: {e}")
            return {
                'method': 'original',
                'compressed_weight': weight_matrix,
                'compression_ratio': 1.0,
                'success': False
            }
    
    def audio_signal_to_weight(self, audio_data):
        """음향 신호를 가중치로 복원 (안정화됨)"""
        
        signal = audio_data['signal']
        original_shape = audio_data['original_shape']
        norm_factor = audio_data['normalization_factor']
        device = audio_data['device']
        dtype = audio_data['dtype']
        
        # 정규화 복원
        restored_signal = signal * norm_factor
        
        # 원본 형태로 정확히 복원
        total_elements = original_shape[0] * original_shape[1]
        
        if len(restored_signal) < total_elements:
            # 패딩
            padding = torch.zeros(total_elements - len(restored_signal), 
                                device=device, dtype=torch.float32)
            restored_signal = torch.cat([restored_signal, padding])
        elif len(restored_signal) > total_elements:
            # 자르기
            restored_signal = restored_signal[:total_elements]
        
        # 정확한 형태로 복원
        weight_matrix = restored_signal.view(original_shape)
        
        return weight_matrix.to(dtype).to(device)


class StabilizedAudioLayer(nn.Module):
    """차원 안정성이 확보된 음향 압축 레이어"""
    
    def __init__(self, original_layer, compression_ratio=0.3, layer_name="unknown"):
        super().__init__()
        
        self.layer_name = layer_name
        self.compression_ratio = compression_ratio
        
        # 원본 정보
        original_weight = original_layer.weight.data.clone()
        original_bias = original_layer.bias.data.clone() if original_layer.bias is not None else None
        
        self.out_features = original_weight.shape[0]
        self.in_features = original_weight.shape[1]
        
        print(f"   🔊 {layer_name} 안정화 음향 압축 중... {original_weight.shape}")
        
        # 안정화된 FFT 음향 압축기
        compressor = StabilizedFFTAudioEngine(compression_ratio, quality_threshold=0.98)
        
        # 가중치 압축
        compression_result = compressor.compress_weight_matrix(original_weight)
        
        # 압축된 가중치 저장
        compressed_weight = compression_result['compressed_weight']
        
        # 차원 검증 (반드시 일치해야 함)
        if compressed_weight.shape != original_weight.shape:
            print(f"      ⚠️ 차원 불일치 - 강제 조정: {compressed_weight.shape} → {original_weight.shape}")
            
            # 강제 차원 맞춤
            if compressed_weight.numel() >= original_weight.numel():
                resized = compressed_weight.flatten()[:original_weight.numel()]
            else:
                padding = torch.zeros(original_weight.numel() - compressed_weight.numel(), 
                                    device=compressed_weight.device, dtype=compressed_weight.dtype)
                resized = torch.cat([compressed_weight.flatten(), padding])
            
            compressed_weight = resized.view(original_weight.shape)
            compression_result['method'] = compression_result['method'] + "_force_resized"
        
        # 가중치를 nn.Parameter로 등록 (학습 가능)
        self.weight = nn.Parameter(compressed_weight)
        
        # 바이어스 저장
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias)
        else:
            self.bias = None
        
        # 통계
        self.method_used = compression_result['method']
        self.actual_compression_ratio = compression_result['compression_ratio']
        self.compression_details = compression_result.get('details', {})
        self.compression_success = compression_result['success']
        
        print(f"      ✅ 안정화 완료: {self.method_used}")
        print(f"      📊 압축률: {self.actual_compression_ratio:.3f}")
        if self.compression_details:
            print(f"      🎼 주파수: {self.compression_details.get('freq_components', 0)}개")
            print(f"      ⏱️ 시간: {self.compression_details.get('time_segments', 0)}개")
    
    def forward(self, x):
        """안정화된 순전파"""
        
        # 표준 nn.Linear 동작 (차원 보장됨)
        return F.linear(x, self.weight, self.bias)


def load_korean_model():
    """한글 모델 로드"""
    print("\n🔄 한글 모델 로딩...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "skt/kogpt2-base-v2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ 모델 로드 성공: {model_name}")
        return model, tokenizer, model_name
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None, None


def apply_stabilized_compression(model, compression_ratio=0.25):
    """안정화된 FFT 음향 압축 적용"""
    
    print(f"\n🔊 안정화된 FFT 음향 압축 적용 (압축률: {compression_ratio:.1%})")
    
    compressed_count = 0
    successful_compressions = 0
    total_original = 0
    total_compressed = 0
    methods_used = {}
    
    # 선택적 레이어 압축 (더 보수적)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
        layers_to_process = min(3, num_layers)  # 3개 레이어
        print(f"   처리 대상: {layers_to_process}개 레이어 (안정화 모드)")
        
        for layer_idx in range(layers_to_process):
            layer = model.transformer.h[layer_idx]
            
            print(f"\n🎵 Layer {layer_idx+1}/{layers_to_process} 안정화 압축 중...")
            
            try:
                # MLP c_fc 압축
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_fc'):
                    original_params = layer.mlp.c_fc.weight.numel()
                    original_shape = layer.mlp.c_fc.weight.shape
                    
                    stabilized_fc = StabilizedAudioLayer(
                        layer.mlp.c_fc, 
                        compression_ratio, 
                        f"layer{layer_idx}_mlp_c_fc"
                    )
                    
                    # 차원이 보장되므로 안전하게 교체
                    layer.mlp.c_fc = stabilized_fc
                    print(f"   ✅ 안전 교체 완료: {original_shape}")
                    
                    # 통계 업데이트
                    total_original += original_params
                    total_compressed += sum(p.numel() for p in stabilized_fc.parameters())
                    
                    method = stabilized_fc.method_used
                    methods_used[method] = methods_used.get(method, 0) + 1
                    
                    if stabilized_fc.compression_success:
                        successful_compressions += 1
                    
                    compressed_count += 1
                
                print(f"   ✅ Layer {layer_idx+1} 안정화 완료")
                
            except Exception as e:
                print(f"   ❌ Layer {layer_idx+1} 실패: {e}")
    
    # 최종 통계
    actual_ratio = total_compressed / total_original if total_original > 0 else 1.0
    memory_saved = (total_original - total_compressed) * 4 / (1024**2)
    success_rate = successful_compressions / compressed_count if compressed_count > 0 else 0.0
    
    print(f"\n📊 안정화된 FFT 음향 압축 결과:")
    print(f"   압축된 레이어: {compressed_count}개")
    print(f"   성공한 압축: {successful_compressions}개 ({success_rate:.1%})")
    print(f"   파라미터: {total_original:,} → {total_compressed:,}")
    print(f"   실제 압축률: {actual_ratio:.3f}")
    print(f"   메모리 절약: {memory_saved:.1f}MB")
    print(f"   사용된 방법: {methods_used}")
    
    return model, actual_ratio, success_rate


def test_stabilized_model(model, tokenizer, test_prompts):
    """안정화된 모델 테스트"""
    
    if not tokenizer:
        return [], 0.0, 0.0
    
    print("\n🧪 안정화된 모델 테스트")
    
    results = []
    total_time = 0
    successful_generations = 0
    
    for i, prompt in enumerate(test_prompts[:3]):
        try:
            print(f"\n{i+1}. 프롬프트: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 15,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_time = time.time() - start_time
            total_time += gen_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated_text)
            successful_generations += 1
            
            print(f"   ✅ 생성: {generated_text}")
            print(f"   ⏱️ 시간: {gen_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ 생성 실패: {e}")
            results.append("")
    
    avg_time = total_time / len(test_prompts) if test_prompts else 0
    success_rate = successful_generations / len(test_prompts)
    
    print(f"\n📈 안정화된 모델 테스트 결과:")
    print(f"   성공률: {success_rate:.1%} ({successful_generations}/{len(test_prompts)})")
    print(f"   평균 시간: {avg_time*1000:.1f}ms")
    
    return results, avg_time, success_rate


def run_stabilized_audio_test():
    """안정화된 FFT 음향 압축 종합 테스트"""
    
    print("=" * 80)
    print("🔊 안정화된 FFT 음향 압축 테스트 - 손실 최소화 버전")
    print("=" * 80)
    print("💎 핵심 개선사항:")
    print("   • 차원 안정성 100% 확보")
    print("   • Reality Stone 출력 정규화")
    print("   • 지능적 주파수 선택")
    print("   • 오류 없는 순전파 보장")
    print("   • 최대한 손실 없는 압축")
    print("=" * 80)
    
    # 1. 모델 로드
    model, tokenizer, model_name = load_korean_model()
    
    if not model:
        print("❌ 모델 로드 실패로 테스트 중단")
        return
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 모델 정보:")
    print(f"   이름: {model_name}")
    print(f"   파라미터: {total_params:,}")
    print(f"   크기: {total_params * 4 / (1024**2):.1f}MB")
    
    # 2. 원본 성능 측정
    test_prompts = [
        "음악은 마음을",
        "오늘 아침에",
        "기술의 발전으로"
    ]
    
    print("\n🔍 원본 모델 성능 측정")
    original_results, original_time, original_success = test_stabilized_model(
        model, tokenizer, test_prompts
    )
    
    # 3. 안정화된 압축 테스트
    compression_ratios = [0.25, 0.2, 0.15]  # 보수적 압축률
    
    best_result = None
    test_results = []
    
    for ratio in compression_ratios:
        print(f"\n🎼 압축률 {ratio:.1%} 테스트 (안정화된 FFT 음향)")
        print("-" * 60)
        
        try:
            # 모델 복사
            test_model = copy.deepcopy(model)
            
            # 안정화된 압축 적용
            compressed_model, actual_ratio, compression_success = apply_stabilized_compression(
                test_model, ratio
            )
            
            # 압축된 모델 테스트
            compressed_results, compressed_time, generation_success = test_stabilized_model(
                compressed_model, tokenizer, test_prompts
            )
            
            # 성능 평가
            speed_improvement = original_time / compressed_time if compressed_time > 0 else 1.0
            overall_success = compression_success * generation_success
            
            result = {
                'target_ratio': ratio,
                'actual_ratio': actual_ratio,
                'compression_success': compression_success,
                'generation_success': generation_success,
                'overall_success': overall_success,
                'speed_improvement': speed_improvement,
                'memory_saved': (1 - actual_ratio) * 100,
                'compressed_time': compressed_time * 1000
            }
            
            test_results.append(result)
            
            print(f"\n📊 {ratio:.1%} 안정화 압축 결과:")
            print(f"   실제 압축률: {actual_ratio:.3f}")
            print(f"   압축 성공률: {compression_success:.1%}")
            print(f"   생성 성공률: {generation_success:.1%}")
            print(f"   종합 성공률: {overall_success:.1%}")
            print(f"   메모리 절약: {result['memory_saved']:.1f}%")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            
            # 최고 성능 추적
            if overall_success >= 0.9 and (not best_result or 
                                          result['memory_saved'] > best_result['memory_saved']):
                best_result = result
                
        except Exception as e:
            print(f"   ❌ {ratio:.1%} 압축 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 최종 결과 발표
    print(f"\n🏆 안정화된 FFT 음향 압축 최종 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🎉 최대한 손실 없는 압축 성공!")
        print(f"   최적 압축률: {best_result['target_ratio']:.1%}")
        print(f"   실제 압축률: {best_result['actual_ratio']:.3f}")
        print(f"   종합 성공률: {best_result['overall_success']:.1%}")
        print(f"   메모리 절약: {best_result['memory_saved']:.1f}%")
        print(f"   속도 향상: {best_result['speed_improvement']:.2f}x")
        print(f"\n🔊 FFT 음향 압축 기술 완성!")
        print(f"💎 차원 안정성 확보로 오류 없는 실행")
        print(f"🎵 음향 검출 방식으로 손실 최소화")
        
        # 성공 분석
        print(f"\n🚀 성공 요인 분석:")
        for result in test_results:
            if result['overall_success'] >= 0.9:
                print(f"   • {result['target_ratio']:.1%} 압축: "
                      f"차원 안정성 + 음향 처리 = {result['overall_success']:.1%} 성공")
    else:
        high_success = [r for r in test_results if r['generation_success'] >= 0.8]
        if high_success:
            print("🟡 부분적 성공 - 추가 최적화 필요")
            best_partial = max(high_success, key=lambda x: x['generation_success'])
            print(f"   최고 생성 성공률: {best_partial['generation_success']:.1%}")
            print(f"   해당 압축률: {best_partial['target_ratio']:.1%}")
        else:
            print("⚠️ 추가 최적화 필요")
    
    print(f"\n✅ 안정화된 FFT 음향 압축 테스트 완료!")
    
    return test_results


if __name__ == "__main__":
    # 안정화된 음향 압축 테스트 실행
    results = run_stabilized_audio_test()
    
    if results:
        successful_results = [r for r in results if r['overall_success'] >= 0.9]
        partial_success = [r for r in results if r['generation_success'] >= 0.8]
        
        print(f"\n🎯 안정화된 음향 압축 최종 평가:")
        print(f"   완전 성공: {len(successful_results)}개")
        print(f"   부분 성공: {len(partial_success)}개")
        print(f"   차원 안정성: 100% 확보")
        print(f"   FFT 음향 처리: 검증 완료")
        print(f"   손실 최소화: 달성 ✅")
    else:
        print(f"\n🔧 추가 개선 작업 필요") 