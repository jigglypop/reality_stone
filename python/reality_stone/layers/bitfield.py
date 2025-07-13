# python/reality_stone/layers/bitfield.py

"""비트필드 기반 하이퍼볼릭 신경망 레이어"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from reality_stone._rust import BitfieldLinear as RustBitfieldLinear


class BitfieldLinearFunction(Function):
    """
    Rust로 구현된 비트필드 직접 추론 커널을 위한 autograd 함수.
    """

    @staticmethod
    def forward(ctx, input, layer_instance):
        """
        순전파에서는 Rust 커널을 직접 호출합니다.
        """
        # autograd 그래프를 통해 역전파가 가능하도록 layer_instance를 저장합니다.
        ctx.layer_instance = layer_instance
        ctx.input_shape = input.shape
        
        # 입력 텐서를 contiguous numpy 배열로 변환하여 Rust 함수에 전달합니다.
        input_np = input.contiguous().detach().cpu().numpy()
        
        # 입력 텐서 차원에 따라 적절한 Rust 함수 호출
        if input.ndim == 2:
            # 2D 텐서: 기존 forward 함수 사용
            output_np = layer_instance.rust_layer.forward(input_np)
        else:
            # 3D 이상 텐서: 다차원 지원 forward_nd 함수 사용
            output_np = layer_instance.rust_layer.forward_nd(input_np)
        
        # 결과를 다시 PyTorch 텐서로 변환합니다.
        output = torch.from_numpy(output_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파에서는 Rust로 구현된 backward 함수를 호출하여
        입력에 대한 그래디언트를 계산합니다.
        """
        # 순전파에서 저장된 layer_instance를 가져옵니다.
        layer_instance = ctx.layer_instance
        input_shape = ctx.input_shape
        
        # grad_output을 numpy 배열로 변환합니다.
        grad_output_np = grad_output.contiguous().detach().cpu().numpy()
        
        # 입력 텐서 차원에 따라 적절한 Rust 함수 호출
        if len(input_shape) == 2:
            # 2D 텐서: 기존 backward 함수 사용
            grad_input_np = layer_instance.rust_layer.backward(grad_output_np)
        else:
            # 3D 이상 텐서: 다차원 지원 backward_nd 함수 사용
            grad_input_np = layer_instance.rust_layer.backward_nd(grad_output_np)
        
        # 결과를 PyTorch 텐서로 변환하여 반환합니다.
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        
        return grad_input, None


class BitfieldLinear(nn.Module):
    """
    비트필드 기반 압축 선형 레이어.
    
    가중치를 22비트 비트필드로 압축하여 저장하고, 
    직접 추론 커널로 순전파를 수행합니다.
    """

    def __init__(self, in_features: int, out_features: int, 
                 basis_size: int = 256, r_max: float = 1.0,
                 use_bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.basis_size = basis_size
        self.r_max = r_max
        
        # Rust 백엔드 레이어 초기화
        self.rust_layer = RustBitfieldLinear(out_features, in_features, basis_size, r_max)
        
        # 바이어스 파라미터 (선택적)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, basis_size: int = 256, r_max: float = 1.0):
        """
        기존 nn.Linear 레이어를 BitfieldLinear로 변환합니다.
        
        # 인자
            linear: 변환할 nn.Linear 레이어
            basis_size: 기저 벡터 테이블 크기 (기본값: 256)
            r_max: 최대 반지름 값 (기본값: 1.0)
        
        # 반환
            변환된 BitfieldLinear 레이어
        """
        use_bias = linear.bias is not None
        
        # 가중치를 numpy 배열로 변환
        weight_np = linear.weight.detach().cpu().numpy()
        
        # Rust 백엔드로 가중치 압축
        rust_layer = RustBitfieldLinear.from_weights(weight_np, basis_size, r_max)
        
        # 새로운 BitfieldLinear 생성
        bitfield_layer = cls(linear.in_features, linear.out_features, 
                           basis_size, r_max, use_bias)
        
        # 압축된 Rust 레이어 할당
        bitfield_layer.rust_layer = rust_layer
        
        # 바이어스 복사
        if use_bias and linear.bias is not None:
            bitfield_layer.bias.data = linear.bias.detach().clone()
        
        return bitfield_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파를 수행합니다.

        # 인자
            input: `[batch_size, in_features]` 또는 `[batch_size, seq_len, in_features]` 형태의 입력 텐서.

        # 반환
            `[batch_size, out_features]` 또는 `[batch_size, seq_len, out_features]` 형태의 출력 텐서.
        """
        # 입력 차원 확인
        if x.ndim < 2:
            raise ValueError(f"입력 텐서는 최소 2차원이어야 합니다. 현재: {x.ndim}차원")
        
        # 마지막 차원이 in_features와 일치하는지 확인
        if x.shape[-1] != self.in_features:
            raise ValueError(f"입력 텐서의 마지막 차원은 {self.in_features}이어야 합니다. 현재: {x.shape[-1]}")
        
        # BitfieldLinearFunction을 통해 순전파 수행
        output = BitfieldLinearFunction.apply(x, self)
        
        # 바이어스 추가
        if self.bias is not None:
            output = output + self.bias
            
        return output

    def __repr__(self):
        return (f"BitfieldLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, basis_size={self.basis_size}, "
                f"bias={self.bias is not None})") 