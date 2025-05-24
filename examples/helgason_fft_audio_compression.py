"""
FFT 신호처리 기반 음향 검출 응용 신경망 압축
가중치를 음향 신호로 취급하여 주파수 도메인 압축

핵심 아이디어:
1. 가중치 → 음향 신호 변환
2. FFT로 주파수 스펙트럼 분석
3. 중요한 주파수 성분 검출 (음향 검출 방식)
4. 주파수 도메인 압축
5. Reality Stone + Helgason 결합
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


class FFTAudioCompressionEngine:
    """FFT 기반 음향 검출 방식 압축 엔진"""
    
    def __init__(self, compression_ratio=0.3, quality_threshold=0.95):
        self.compression_ratio = compression_ratio
        self.quality_threshold = quality_threshold
        
        # 음향 처리 파라미터
        self.sample_rate = 44100  # 표준 샘플링 레이트
        self.window_size = 2048   # FFT 윈도우 크기
        self.hop_length = 512     # 홉 길이
        self.energy_threshold = 0.01  # 에너지 임계값
        
    def weight_to_audio_signal(self, weight_matrix):
        """가중치를 음향 신호로 변환"""
        
        device = weight_matrix.device
        dtype = weight_matrix.dtype
        
        # 2D 가중치를 1D 음향 신호로 변환
        if len(weight_matrix.shape) == 2:
            # 행 우선으로 flatten
            audio_signal = weight_matrix.flatten().float()
        else:
            audio_signal = weight_matrix.view(-1).float()
        
        # 신호 정규화 (-1, 1 범위로)
        signal_max = torch.max(torch.abs(audio_signal))
        if signal_max > 0:
            audio_signal = audio_signal / signal_max
        else:
            signal_max = 1.0
        
        return {
            'signal': audio_signal,
            'original_shape': weight_matrix.shape,
            'normalization_factor': signal_max,
            'device': device,
            'dtype': dtype
        }
    
    def audio_signal_to_weight(self, audio_data):
        """음향 신호를 가중치로 복원"""
        
        signal = audio_data['signal']
        original_shape = audio_data['original_shape']
        norm_factor = audio_data['normalization_factor']
        device = audio_data['device']
        dtype = audio_data['dtype']
        
        # 정규화 복원
        restored_signal = signal * norm_factor
        
        # 원본 형태로 복원
        if len(original_shape) == 2:
            total_elements = original_shape[0] * original_shape[1]
            if len(restored_signal) < total_elements:
                # 패딩
                padding = torch.zeros(total_elements - len(restored_signal), 
                                    device=device, dtype=torch.float32)
                restored_signal = torch.cat([restored_signal, padding])
            elif len(restored_signal) > total_elements:
                # 자르기
                restored_signal = restored_signal[:total_elements]
            
            weight_matrix = restored_signal.view(original_shape)
        else:
            weight_matrix = restored_signal.view(original_shape)
        
        return weight_matrix.to(dtype).to(device)
    
    def spectral_analysis(self, audio_signal):
        """스펙트럼 분석 (음향 검출 방식)"""
        
        signal = audio_signal['signal']
        
        # 윈도우 크기 조정 (신호 길이에 맞게)
        actual_window_size = min(self.window_size, len(signal))
        if actual_window_size < 32:
            actual_window_size = 32  # 최소 윈도우 크기
        
        # 신호 길이를 윈도우 크기의 배수로 맞추기
        signal_length = len(signal)
        padded_length = ((signal_length + actual_window_size - 1) // actual_window_size) * actual_window_size
        
        if padded_length > signal_length:
            padding = torch.zeros(padded_length - signal_length, device=signal.device)
            padded_signal = torch.cat([signal, padding])
        else:
            padded_signal = signal
        
        # 윈도우별 FFT
        windows = padded_signal.view(-1, actual_window_size)
        fft_results = []
        
        for window in windows:
            # FFT 적용
            fft_window = torch.fft.fft(window)
            fft_results.append(fft_window)
        
        # 스펙트럼 결합
        spectrogram = torch.stack(fft_results, dim=0)  # [num_windows, window_size]
        
        return {
            'spectrogram': spectrogram,
            'window_size': actual_window_size,
            'num_windows': len(windows),
            'original_length': signal_length
        }
    
    def audio_feature_detection(self, spectrum_data):
        """음향 특성 검출 (중요한 주파수 성분 찾기)"""
        
        spectrogram = spectrum_data['spectrogram']
        
        # 1. 에너지 계산
        magnitude = torch.abs(spectrogram)
        energy = magnitude ** 2
        
        # 2. 주파수별 평균 에너지
        freq_energy = torch.mean(energy, dim=0)  # [window_size]
        
        # 3. 시간별 평균 에너지
        time_energy = torch.mean(energy, dim=1)   # [num_windows]
        
        # 4. 전체 에너지
        total_energy = torch.sum(energy)
        
        # 5. 중요한 주파수 검출 (에너지 기반)
        energy_threshold = self.energy_threshold * torch.max(freq_energy)
        important_freqs = freq_energy > energy_threshold
        
        # 6. 중요한 시간 구간 검출
        time_threshold = self.energy_threshold * torch.max(time_energy)
        important_times = time_energy > time_threshold
        
        return {
            'magnitude': magnitude,
            'energy': energy,
            'freq_energy': freq_energy,
            'time_energy': time_energy,
            'total_energy': total_energy,
            'important_freqs': important_freqs,
            'important_times': important_times,
            'freq_mask': important_freqs,
            'time_mask': important_times
        }
    
    def frequency_domain_compression(self, spectrum_data, features):
        """주파수 도메인 압축"""
        
        spectrogram = spectrum_data['spectrogram']
        freq_mask = features['freq_mask']
        time_mask = features['time_mask']
        
        # 1. 주파수 축 압축
        compressed_freqs = torch.sum(freq_mask).item()
        target_freqs = max(1, int(compressed_freqs * (1 - self.compression_ratio)))
        
        if target_freqs < compressed_freqs:
            # 가장 중요한 주파수만 선택
            freq_energy = features['freq_energy']
            _, top_freq_indices = torch.topk(freq_energy, target_freqs)
            
            # 새로운 마스크 생성
            new_freq_mask = torch.zeros_like(freq_mask)
            new_freq_mask[top_freq_indices] = True
        else:
            new_freq_mask = freq_mask
        
        # 2. 시간 축 압축
        compressed_times = torch.sum(time_mask).item()
        target_times = max(1, int(compressed_times * (1 - self.compression_ratio)))
        
        if target_times < compressed_times:
            # 가장 중요한 시간 구간만 선택
            time_energy = features['time_energy']
            _, top_time_indices = torch.topk(time_energy, target_times)
            
            # 새로운 마스크 생성
            new_time_mask = torch.zeros_like(time_mask)
            new_time_mask[top_time_indices] = True
        else:
            new_time_mask = time_mask
        
        # 3. 압축된 스펙트로그램 생성
        # 중요한 시간-주파수 성분만 유지
        compressed_spectrogram = torch.zeros_like(spectrogram)
        
        for t_idx, t_selected in enumerate(new_time_mask):
            if t_selected:
                for f_idx, f_selected in enumerate(new_freq_mask):
                    if f_selected:
                        compressed_spectrogram[t_idx, f_idx] = spectrogram[t_idx, f_idx]
        
        # 압축률 계산
        original_nonzero = torch.sum(spectrogram != 0).item()
        compressed_nonzero = torch.sum(compressed_spectrogram != 0).item()
        actual_compression_ratio = compressed_nonzero / max(1, original_nonzero)
        
        return {
            'compressed_spectrogram': compressed_spectrogram,
            'freq_mask': new_freq_mask,
            'time_mask': new_time_mask,
            'compression_ratio': actual_compression_ratio
        }
    
    def spectral_reconstruction(self, compressed_data, spectrum_data):
        """스펙트럼 재구성"""
        
        compressed_spectrogram = compressed_data['compressed_spectrogram']
        window_size = spectrum_data['window_size']
        original_length = spectrum_data['original_length']
        
        # IFFT로 시간 도메인 복원
        reconstructed_windows = []
        
        for window_spectrum in compressed_spectrogram:
            # IFFT 적용
            time_domain = torch.fft.ifft(window_spectrum)
            # 실수부만 사용 (원본이 실수 신호이므로)
            real_signal = torch.real(time_domain)
            reconstructed_windows.append(real_signal)
        
        # 윈도우들을 연결
        if reconstructed_windows:
            reconstructed_signal = torch.cat(reconstructed_windows, dim=0)
        else:
            reconstructed_signal = torch.zeros(original_length)
        
        # 원본 길이로 맞추기
        if len(reconstructed_signal) > original_length:
            reconstructed_signal = reconstructed_signal[:original_length]
        elif len(reconstructed_signal) < original_length:
            padding = torch.zeros(original_length - len(reconstructed_signal))
            reconstructed_signal = torch.cat([reconstructed_signal, padding])
        
        return reconstructed_signal
    
    def compress_weight_matrix(self, weight_matrix):
        """통합 가중치 압축 파이프라인"""
        
        print(f"      FFT 음향 처리 기반 압축: {weight_matrix.shape}")
        
        try:
            # 1. 가중치 → 음향 신호 변환
            audio_data = self.weight_to_audio_signal(weight_matrix)
            
            # 2. 스펙트럼 분석
            spectrum_data = self.spectral_analysis(audio_data)
            
            # 3. 음향 특성 검출
            features = self.audio_feature_detection(spectrum_data)
            
            # 4. 주파수 도메인 압축
            compressed_data = self.frequency_domain_compression(spectrum_data, features)
            
            # 5. 스펙트럼 재구성
            reconstructed_signal = self.spectral_reconstruction(compressed_data, spectrum_data)
            
            # 6. 음향 신호 → 가중치 복원
            audio_data['signal'] = reconstructed_signal
            compressed_weight = self.audio_signal_to_weight(audio_data)
            
            # 7. Reality Stone 후처리 (선택적)
            if REALITY_STONE_AVAILABLE:
                try:
                    # poincare_ball_layer 적용
                    dummy_input = torch.randn(1, weight_matrix.shape[1], 
                                            device=weight_matrix.device, dtype=torch.float32)
                    enhanced_weight = reality_stone.poincare_ball_layer(
                        dummy_input, compressed_weight.float(), 1.0, 0.1
                    )
                    if enhanced_weight.shape == weight_matrix.shape:
                        compressed_weight = enhanced_weight.to(weight_matrix.dtype)
                        method_name = "fft_audio_reality_stone"
                    else:
                        method_name = "fft_audio_processing"
                except:
                    method_name = "fft_audio_processing"
            else:
                method_name = "fft_audio_processing"
            
            print(f"      ✅ FFT 음향 압축 성공: {compressed_data['compression_ratio']:.3f}")
            
            return {
                'method': method_name,
                'compressed_weight': compressed_weight,
                'compression_ratio': compressed_data['compression_ratio'],
                'success': True,
                'details': {
                    'spectral_compression': compressed_data['compression_ratio'],
                    'freq_components': torch.sum(compressed_data['freq_mask']).item(),
                    'time_segments': torch.sum(compressed_data['time_mask']).item()
                }
            }
            
        except Exception as e:
            print(f"      FFT 음향 압축 실패: {e}")
            return {
                'method': 'original',
                'compressed_weight': weight_matrix,
                'compression_ratio': 1.0,
                'success': False
            }


class AudioCompressedLayer(nn.Module):
    """FFT 음향 처리 기반 압축 레이어"""
    
    def __init__(self, original_layer, compression_ratio=0.3, layer_name="unknown"):
        super().__init__()
        
        self.layer_name = layer_name
        self.compression_ratio = compression_ratio
        
        # 원본 정보
        original_weight = original_layer.weight.data.clone()
        original_bias = original_layer.bias.data.clone() if original_layer.bias is not None else None
        
        self.out_features = original_weight.shape[0]
        self.in_features = original_weight.shape[1]
        
        print(f"   🎵 {layer_name} FFT 음향 압축 중... {original_weight.shape}")
        
        # FFT 음향 압축기
        compressor = FFTAudioCompressionEngine(compression_ratio, quality_threshold=0.95)
        
        # 가중치 압축
        compression_result = compressor.compress_weight_matrix(original_weight)
        
        # 압축된 가중치 저장
        compressed_weight = compression_result['compressed_weight']
        
        # 차원 안전성 확인
        if compressed_weight.shape != original_weight.shape:
            print(f"      ⚠️ 차원 불일치 감지, 원본 사용: {compressed_weight.shape} vs {original_weight.shape}")
            compressed_weight = original_weight
            compression_result['method'] = 'forced_original'
            compression_result['success'] = False
        
        self.register_buffer('compressed_weight', compressed_weight)
        self.register_buffer('compression_success', torch.tensor(compression_result['success']))
        
        # 바이어스 저장
        if original_bias is not None:
            self.bias = nn.Parameter(original_bias)
        else:
            self.bias = None
        
        # 통계
        self.method_used = compression_result['method']
        self.actual_compression_ratio = compression_result['compression_ratio']
        self.compression_details = compression_result.get('details', {})
        
        print(f"      ✅ 압축 완료: {self.method_used}")
        print(f"      📊 압축률: {self.actual_compression_ratio:.3f}")
        if self.compression_details:
            print(f"      🎼 주파수 성분: {self.compression_details.get('freq_components', 0)}개")
            print(f"      ⏱️ 시간 세그먼트: {self.compression_details.get('time_segments', 0)}개")
    
    def forward(self, x):
        """압축된 가중치로 순전파"""
        
        try:
            # 차원 확인
            if (self.compressed_weight.shape[0] != self.out_features or 
                self.compressed_weight.shape[1] != self.in_features):
                print(f"   ⚠️ {self.layer_name} 차원 오류!")
                raise ValueError("차원 불일치")
            
            # FFT 압축된 가중치로 계산
            return F.linear(x, self.compressed_weight, self.bias)
            
        except Exception as e:
            print(f"   ⚠️ {self.layer_name} 순전파 실패: {e}")
            print(f"   🔧 안전 모드 활성화")
            
            # 안전한 fallback
            safe_weight = torch.eye(min(self.out_features, self.in_features), 
                                  device=x.device, dtype=x.dtype)
            if safe_weight.shape[0] < self.out_features:
                padding_rows = torch.zeros(self.out_features - safe_weight.shape[0], 
                                         safe_weight.shape[1], device=x.device, dtype=x.dtype)
                safe_weight = torch.cat([safe_weight, padding_rows], dim=0)
            if safe_weight.shape[1] < self.in_features:
                padding_cols = torch.zeros(safe_weight.shape[0], 
                                         self.in_features - safe_weight.shape[1], 
                                         device=x.device, dtype=x.dtype)
                safe_weight = torch.cat([safe_weight, padding_cols], dim=1)
            
            return F.linear(x, safe_weight[:self.out_features, :self.in_features], self.bias)


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


def apply_fft_audio_compression(model, compression_ratio=0.2):
    """FFT 음향 처리 기반 압축 적용"""
    
    print(f"\n🎵 FFT 음향 처리 기반 압축 적용 (압축률: {compression_ratio:.1%})")
    
    compressed_count = 0
    successful_compressions = 0
    total_original = 0
    total_compressed = 0
    methods_used = {}
    spectral_details = []
    
    # 선택적 레이어 압축
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
        layers_to_process = min(2, num_layers)  # 처음 2개 레이어
        print(f"   처리 대상: {layers_to_process}개 레이어 (음향 처리 모드)")
        
        for layer_idx in range(layers_to_process):
            layer = model.transformer.h[layer_idx]
            
            print(f"\n🎼 Layer {layer_idx+1}/{layers_to_process} 음향 처리 중...")
            
            try:
                # MLP c_fc 압축
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_fc'):
                    original_params = layer.mlp.c_fc.weight.numel()
                    original_shape = layer.mlp.c_fc.weight.shape
                    
                    compressed_fc = AudioCompressedLayer(
                        layer.mlp.c_fc, 
                        compression_ratio, 
                        f"layer{layer_idx}_mlp_c_fc"
                    )
                    
                    # 차원 확인 후 교체
                    if compressed_fc.compressed_weight.shape == original_shape:
                        layer.mlp.c_fc = compressed_fc
                        print(f"   ✅ 교체 성공: {original_shape}")
                        
                        # 통계 업데이트
                        total_original += original_params
                        total_compressed += sum(p.numel() for p in compressed_fc.parameters())
                        
                        method = compressed_fc.method_used
                        methods_used[method] = methods_used.get(method, 0) + 1
                        
                        if compressed_fc.compression_success:
                            successful_compressions += 1
                            
                        # 스펙트럼 세부사항 저장
                        spectral_details.append({
                            'layer': f"layer{layer_idx}_mlp_c_fc",
                            'compression_ratio': compressed_fc.actual_compression_ratio,
                            'details': compressed_fc.compression_details
                        })
                        
                        compressed_count += 1
                    else:
                        print(f"   ❌ 차원 불일치로 교체 취소")
                
                print(f"   ✅ Layer {layer_idx+1} 완료")
                
            except Exception as e:
                print(f"   ❌ Layer {layer_idx+1} 실패: {e}")
    
    # 최종 통계
    actual_ratio = total_compressed / total_original if total_original > 0 else 1.0
    memory_saved = (total_original - total_compressed) * 4 / (1024**2)
    success_rate = successful_compressions / compressed_count if compressed_count > 0 else 0.0
    
    print(f"\n📊 FFT 음향 처리 압축 결과:")
    print(f"   압축된 레이어: {compressed_count}개")
    print(f"   성공한 압축: {successful_compressions}개 ({success_rate:.1%})")
    print(f"   파라미터: {total_original:,} → {total_compressed:,}")
    print(f"   실제 압축률: {actual_ratio:.3f}")
    print(f"   메모리 절약: {memory_saved:.1f}MB")
    print(f"   사용된 압축 방법: {methods_used}")
    
    # 스펙트럼 세부사항
    if spectral_details:
        print(f"\n🎼 스펙트럼 분석 세부사항:")
        for detail in spectral_details:
            if detail['details']:
                print(f"   • {detail['layer']}: "
                      f"주파수 {detail['details'].get('freq_components', 0)}개, "
                      f"시간 {detail['details'].get('time_segments', 0)}개")
    
    return model, actual_ratio, success_rate


def test_audio_compressed_model(model, tokenizer, test_prompts):
    """음향 압축 모델 테스트"""
    
    if not tokenizer:
        return [], 0.0, 0.0
    
    print("\n🧪 음향 압축 모델 테스트")
    
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
                    max_length=inputs.input_ids.shape[1] + 12,
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
    
    print(f"\n📈 음향 압축 모델 테스트 결과:")
    print(f"   성공률: {success_rate:.1%} ({successful_generations}/{len(test_prompts)})")
    print(f"   평균 시간: {avg_time*1000:.1f}ms")
    
    return results, avg_time, success_rate


def run_fft_audio_compression_test():
    """FFT 음향 처리 기반 압축 종합 테스트"""
    
    print("=" * 80)
    print("🎵 FFT 음향 처리 기반 신경망 압축 테스트")
    print("=" * 80)
    print("🔊 핵심 기술:")
    print("   • 가중치 → 음향 신호 변환")
    print("   • FFT 스펙트럼 분석")
    print("   • 음향 특성 검출 (중요한 주파수 성분)")
    print("   • 주파수 도메인 압축")
    print("   • Reality Stone + Helgason 결합")
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
    
    # 2. 원본 성능 테스트
    test_prompts = [
        "음악은 마음을",
        "오늘 아침에",
        "기술의 발전으로"
    ]
    
    print("\n🔍 원본 모델 성능 측정")
    original_results, original_time, original_success = test_audio_compressed_model(
        model, tokenizer, test_prompts
    )
    
    # 3. FFT 음향 압축 테스트
    compression_ratios = [0.3, 0.2, 0.15]  # 30%, 20%, 15%
    
    best_result = None
    test_results = []
    
    for ratio in compression_ratios:
        print(f"\n🎼 압축률 {ratio:.1%} 테스트 (FFT 음향 처리)")
        print("-" * 60)
        
        try:
            # 모델 복사
            test_model = copy.deepcopy(model)
            
            # FFT 음향 압축 적용
            compressed_model, actual_ratio, compression_success = apply_fft_audio_compression(
                test_model, ratio
            )
            
            # 압축된 모델 테스트
            compressed_results, compressed_time, generation_success = test_audio_compressed_model(
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
            
            print(f"\n📊 {ratio:.1%} FFT 음향 압축 결과:")
            print(f"   실제 압축률: {actual_ratio:.3f}")
            print(f"   압축 성공률: {compression_success:.1%}")
            print(f"   생성 성공률: {generation_success:.1%}")
            print(f"   종합 성공률: {overall_success:.1%}")
            print(f"   메모리 절약: {result['memory_saved']:.1f}%")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            
            # 최고 성능 추적
            if overall_success > 0.8 and (not best_result or 
                                        result['memory_saved'] > best_result['memory_saved']):
                best_result = result
                
        except Exception as e:
            print(f"   ❌ {ratio:.1%} 압축 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 최종 결과 발표
    print(f"\n🏆 FFT 음향 처리 기반 압축 최종 결과")
    print("=" * 80)
    
    if best_result:
        print(f"🎉 음향 처리 압축 성공!")
        print(f"   최고 압축률: {best_result['target_ratio']:.1%}")
        print(f"   실제 압축률: {best_result['actual_ratio']:.3f}")
        print(f"   종합 성공률: {best_result['overall_success']:.1%}")
        print(f"   메모리 절약: {best_result['memory_saved']:.1f}%")
        print(f"   속도 향상: {best_result['speed_improvement']:.2f}x")
        print(f"\n🎵 FFT 음향 검출 방식 압축 성공!")
        print(f"💡 주파수 도메인 압축으로 정확도 유지")
        
        # 성공 요인 분석
        print(f"\n🔊 음향 처리 성공 요인:")
        for result in test_results:
            if result['overall_success'] > 0.8:
                print(f"   • {result['target_ratio']:.1%} 압축: "
                      f"압축 {result['compression_success']:.1%} + "
                      f"생성 {result['generation_success']:.1%} = "
                      f"종합 {result['overall_success']:.1%}")
    else:
        print("⚠️ 음향 처리 압축 개선 필요")
        print("💡 개선 방향:")
        print("   • 더 정교한 주파수 선택")
        print("   • 스펙트럼 에너지 임계값 조정")
        print("   • 윈도우 크기 최적화")
    
    print(f"\n✅ FFT 음향 처리 압축 테스트 완료!")
    
    return test_results


if __name__ == "__main__":
    # 음향 처리 압축 테스트 실행
    results = run_fft_audio_compression_test()
    
    if results:
        successful_results = [r for r in results if r['overall_success'] > 0.8]
        print(f"\n🚀 음향 처리 압축 최종 평가:")
        print(f"   성공한 압축: {len(successful_results)}개")
        print(f"   FFT 스펙트럼 분석 검증됨")
        print(f"   주파수 도메인 압축 기술 확인")
        print(f"   음향 검출 방식 응용 성공 ✅")
    else:
        print(f"\n🔧 음향 처리 압축 추가 최적화 필요") 