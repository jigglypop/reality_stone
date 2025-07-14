# python/reality_stone/layers/bitfield.py

"""비트필드 기반 하이퍼볼릭 신경망 레이어"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torch.autograd import Function
from reality_stone._rust import BitfieldLinear as RustBitfieldLinear

# 전역 basis table 캐시 - 모든 레이어가 공유
_BASIS_TABLE_CACHE = {}

class BitfieldLinearFunction(Function):
    """
    Rust의 `tch-rs`를 사용하여 PyTorch 텐서를 직접 처리하는 autograd 함수.
    GPU 텐서는 복사 없이 그대로 Rust로 전달되어 처리됩니다.
    """
    @staticmethod
    def forward(ctx, input, layer_instance):
        ctx.layer_instance = layer_instance
        
        # Rust 레이어가 입력 텐서의 device를 자동으로 감지하여 처리
        output = layer_instance.rust_layer.forward(input)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        layer_instance = ctx.layer_instance
        
        # Rust 레이어가 grad_output의 device를 자동으로 감지하여 처리
        grad_input = layer_instance.rust_layer.backward(grad_output)
        
        return grad_input, None


class BitfieldFunction(Function):
    """GPU 최적화를 위한 커스텀 autograd Function"""
    
    @staticmethod
    def forward(ctx, input, rust_layer, bias):
        ctx.rust_layer = rust_layer
        ctx.save_for_backward(input)
        ctx.has_bias = bias is not None
        
        # GPU 텐서 직접 처리
        if input.is_cuda and rust_layer.use_cuda():
            # GPU 포인터와 메타데이터 추출
            batch_size = input.shape[0]
            in_features = input.shape[-1]
            out_features = rust_layer.get_m()
            
            # 출력 텐서를 미리 할당
            output = torch.empty(
                (*input.shape[:-1], out_features), 
                device=input.device, 
                dtype=input.dtype
            )
            
            # 입력이 2D가 아닌 경우 reshape
            if input.dim() > 2:
                input_2d = input.view(-1, in_features)
                output_2d = output.view(-1, out_features)
            else:
                input_2d = input
                output_2d = output
            
            # GPU 메모리 포인터 직접 전달
            try:
                # GPU 버퍼 초기화 (필요한 경우만)
                try:
                    rust_layer.init_gpu_memory()
                except:
                    pass  # 이미 초기화된 경우 무시
                
                # CUDA 커널 직접 호출 (복사 없음)
                if hasattr(rust_layer, 'forward_cuda_direct'):
                    rust_layer.forward_cuda_direct(
                        input_2d.data_ptr(),
                        output_2d.data_ptr(),
                        batch_size,
                        in_features,
                        out_features
                    )
                else:
                    # forward_cuda_direct가 없으면 기존 방식 사용
                    output = rust_layer.forward(input)
                    # 출력이 CPU로 나왔다면 GPU로 이동
                    if not output.is_cuda and input.is_cuda:
                        output = output.to(input.device)
                    return output
                
            except Exception as e:
                # GPU 처리 실패 시 기존 방식 사용
                print(f"[경고] GPU 직접 처리 실패: {e}")
                output = rust_layer.forward(input)
                # 출력이 CPU로 나왔다면 GPU로 이동
                if not output.is_cuda and input.is_cuda:
                    output = output.to(input.device)
        else:
            # CPU 경로 또는 CUDA 비활성화
            output = rust_layer.forward(input)
        
        # bias 추가 (in-place 연산으로 메모리 효율성 개선)
        if bias is not None:
            output.add_(bias)
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        rust_layer = ctx.rust_layer
        
        grad_input = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # GPU 텐서 직접 처리
            if grad_output.is_cuda and rust_layer.use_cuda():
                batch_size = grad_output.shape[0]
                out_features = grad_output.shape[-1]
                in_features = input.shape[-1]
                
                # 그라디언트 텐서를 미리 할당
                grad_input = torch.empty_like(input)
                
                # 다차원 텐서 처리
                if grad_output.dim() > 2:
                    grad_output_2d = grad_output.view(-1, out_features)
                    grad_input_2d = grad_input.view(-1, in_features)
                else:
                    grad_output_2d = grad_output
                    grad_input_2d = grad_input
                
                try:
                    # CUDA 커널 직접 호출
                    if hasattr(rust_layer, 'backward_cuda_direct'):
                        rust_layer.backward_cuda_direct(
                            grad_output_2d.data_ptr(),
                            grad_input_2d.data_ptr(),
                            batch_size,
                            in_features,
                            out_features
                        )
                    else:
                        grad_input = rust_layer.backward(grad_output)
                except Exception as e:
                    print(f"[경고] GPU backward 실패: {e}")
                    grad_input = rust_layer.backward(grad_output)
            else:
                # CPU 경로
                grad_input = rust_layer.backward(grad_output)
        
        # bias gradient (필요한 경우만)
        if ctx.needs_input_grad[2] and ctx.has_bias:
            # 모든 배치에 대해 합산
            grad_bias = grad_output.sum(dim=list(range(grad_output.dim() - 1)))
        
        return grad_input, None, grad_bias


class BitfieldLinear(nn.Module):
    """
    비트필드 기반 압축 선형 레이어.
    
    가중치를 32비트 비트필드로 압축하여 저장하고, 
    직접 추론 커널로 순전파를 수행합니다.
    
    Args:
        in_features: 입력 특성 수
        out_features: 출력 특성 수
        basis_size: 기저 벡터 테이블 크기 (기본값: 256)
        r_max: 최대 반지름 값 (기본값: 1.0)
        use_bias: 바이어스 사용 여부 (기본값: True)
        use_cuda: CUDA 사용 여부 (기본값: None, 자동 감지)
    """

    def __init__(self, in_features: int, out_features: int, 
                 use_bias: bool = True, basis_size: int = 256, r_max: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.basis_size = basis_size
        self.r_max = r_max
        self.rust_layer = None
        
        # 압축 정보 저장
        self.use_residual = True
        self.compression_level = 1
        self.use_int8 = False
        self.use_tensorcore = False
        self.use_hierarchical = False
        
        # GPU 텐서들 (lazy initialization)
        self.codes_gpu = None
        self.basis_table_gpu = None
        self.residual_codes_gpu = None
        self.residual_int8_gpu = None
        self.residual_scales_gpu = None
        self.delta = None
        self.residual_delta = None
        
        self.rust_layer = RustBitfieldLinear(out_features, in_features, basis_size, r_max)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, basis_size: int = 256, r_max: float = 1.0, 
                   use_residual: bool = True) -> 'BitfieldLinear':
        """기존 Linear 레이어를 BitfieldLinear로 변환합니다.
        
        Args:
            linear: 변환할 nn.Linear 레이어
            basis_size: 기저 벡터 개수 (기본값: 256)
            r_max: 최대 반지름 값 (기본값: 1.0)
            use_cuda: CUDA 사용 여부 (None이면 자동 감지)
            use_residual: 잔차 가중치 사용 여부 (False면 극한 압축)
            
        Returns:
            압축된 BitfieldLinear 레이어
        """
        use_bias = linear.bias is not None
        
        # 가중치를 numpy로 변환
        weight_np = linear.weight.detach().cpu().numpy()
        bias_np = linear.bias.detach().cpu().numpy() if use_bias else None
        
        # Rust BitfieldLinear 생성
        rust_layer = RustBitfieldLinear.from_weights(
            weight_np, bias_np, basis_size, r_max, use_residual
        )
        
        # CUDA 사용 설정
        use_cuda = linear.weight.is_cuda
        rust_layer.set_use_cuda(use_cuda)
        
        if use_cuda:
            print(f"  CUDA 가속 활성화됨 (device: {linear.weight.device})")
            # GPU 메모리 초기화
            rust_layer.init_gpu_memory()
        else:
            print(f"  CPU 모드 (device: {linear.weight.device})")
        
        # 새로운 BitfieldLinear 생성
        bitfield = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            basis_size=basis_size,
            r_max=r_max,
            use_bias=use_bias,
            # use_cuda=use_cuda # This line is removed
        )
        
        bitfield.rust_layer = rust_layer
        
        # 압축 정보 저장
        bitfield.use_residual = use_residual
        
        # 원본 bias 저장 (올바른 device로)
        if use_bias:
            bitfield.bias.data = torch.from_numpy(bias_np).to(linear.weight.device)
        
        # BitfieldLinear를 올바른 device로 이동
        bitfield = bitfield.to(linear.weight.device)
        
        # 압축 정보 출력
        original_size = linear.in_features * linear.out_features * 4 / 1024  # KB
        
        if use_residual:
            # 압축 크기: codes + residuals + scales (basis table은 공유되므로 제외)
            compressed_size = (
                linear.out_features * 4 +  # codes
                linear.out_features * linear.in_features * 1 +  # residual_weights_int8
                linear.out_features * 4  # residual_scales
            ) / 1024  # KB
        else:
            # 극한 압축: codes만 저장
            compressed_size = linear.out_features * 4 / 1024  # KB
        
        print(f"  압축 정보: original {original_size:.2f} KB -> compressed {compressed_size:.2f} KB")
        print(f"  압축률: {original_size / compressed_size:.1f}x")
        if not use_residual:
            print(f"  극한 압축 모드 활성화 (잔차 없음)")
        
        return bitfield
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 새로운 최적화된 BitfieldFunction 사용
        return BitfieldFunction.apply(x, self.rust_layer, self.bias)
    
    def to(self, *args, **kwargs):
        """
        레이어를 특정 장치로 이동시키고, Rust 백엔드의 CUDA 사용 상태를 동기화합니다.
        """
        result = super().to(*args, **kwargs)

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
        elif len(args) > 0 and args[0] is not None:
            device = args[0]
        
        if device is not None:
            device = torch.device(device)
            if device.type == 'cuda':
                self.rust_layer.set_use_cuda(True)
                # GPU로 이동 시 GPU 메모리 초기화
                try:
                    self.rust_layer.init_gpu_memory()
                except Exception as e:
                    print(f"경고: GPU 메모리 초기화 실패: {e}")
                    self.rust_layer.set_use_cuda(False)
            else:
                self.rust_layer.set_use_cuda(False)
        
        return result

    def is_cuda_enabled(self) -> bool:
        """CUDA 사용 여부를 반환합니다."""
        return self.rust_layer.use_cuda()
    
    def enable_int8_optimization(self):
        """INT8 최적화를 활성화합니다."""
        self.rust_layer.enable_int8_optimization()
        self.use_int8 = True
    
    def enable_tensorcore(self):
        """Tensor Core 최적화를 활성화합니다."""
        self.rust_layer.enable_tensorcore()
        self.use_tensorcore = True
    
    def enable_hierarchical_compression(self, level=2):
        """계층적 압축을 활성화합니다.
        
        Args:
            level: 압축 레벨 (2: 4비트/가중치, 3: 2.5비트/가중치)
        """
        self.rust_layer.enable_hierarchical_compression(level)
        self.use_hierarchical = True
        self.compression_level = level
    
    def enable_qat(self, temperature=1.0):
        """Quantization-Aware Training을 활성화합니다.
        
        Args:
            temperature: Gumbel-Softmax 온도
        """
        self.rust_layer.enable_qat(temperature)
    
    def anneal_temperature(self, decay_rate=0.95):
        """QAT 온도를 점진적으로 낮춥니다.
        
        Args:
            decay_rate: 온도 감소율
        """
        self.rust_layer.anneal_temperature(decay_rate)
    
    def quantization_loss(self):
        """양자화 손실을 반환합니다."""
        return self.rust_layer.quantization_loss()
    
    def get_compression_ratio(self):
        """현재 설정에 따른 압축률을 계산합니다."""
        original_size = self.in_features * self.out_features * 4  # FP32 bytes
        
        # 기본 코드 크기
        compressed_size = self.out_features * 4  # 32비트 코드
        
        # 잔차 추가
        if self.use_residual:
            compressed_size += self.out_features * self.in_features * 1  # INT8 잔차
            compressed_size += self.out_features * 4  # 스케일
        
        # 계층적 압축
        if self.use_hierarchical:
            if self.compression_level == 2:
                # 4비트/가중치
                compressed_size = self.out_features * 0.5
            elif self.compression_level == 3:
                # 2.5비트/가중치  
                compressed_size = self.out_features * 0.3125
        
        # 기저 테이블 (공유되므로 출력 수로 나눔)
        basis_table_size = self.basis_size * self.in_features * 4
        if self.use_int8:
            basis_table_size = self.basis_size * self.in_features * 1  # INT8
        compressed_size += basis_table_size / max(self.out_features, 1)
        
        return original_size / compressed_size
    
    def __repr__(self):
        return (f"BitfieldLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, use_cuda={self.rust_layer.use_cuda()})") 