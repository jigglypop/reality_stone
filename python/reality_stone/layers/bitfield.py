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
        
        # 입력 텐서를 contiguous numpy 배열로 변환하여 Rust 함수에 전달합니다.
        input_np = input.contiguous().detach().cpu().numpy()
        
        # Rust의 forward 함수를 호출합니다.
        output_np = layer_instance.rust_layer.forward(input_np)
        
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
        
        # grad_output을 numpy 배열로 변환합니다.
        grad_output_np = grad_output.contiguous().detach().cpu().numpy()
        
        # Rust의 backward 함수를 호출합니다.
        grad_input_np = layer_instance.rust_layer.backward(grad_output_np)
        
        # 결과를 PyTorch 텐서로 변환하여 반환합니다.
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        
        # forward 함수의 입력은 (input, layer_instance) 였으므로,
        # 그래디언트도 두 개를 반환해야 합니다. layer_instance에 대한 그래디언트는 None입니다.
        return grad_input, None


class BitfieldLinear(nn.Module):
    """
    비트필드 인코딩을 사용한 선형 레이어
    
    가중치를 22비트로 압축하여 메모리 사용량을 대폭 줄이면서도
    하이퍼볼릭 기하학을 활용해 표현력을 유지합니다.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 basis_size: int = 256, r_max: float = 1.0):
        """
        Args:
            in_features: 입력 피처 수
            out_features: 출력 피처 수  
            basis_size: 기저 벡터 테이블 크기 (기본값: 256)
            r_max: 최대 반지름 값 (기본값: 1.0)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_size = basis_size
        self.r_max = r_max
        
        # Rust 구현 초기화
        self.rust_layer = RustBitfieldLinear(
            out_features, in_features, basis_size, r_max
        )
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, basis_size: int = 256, r_max: float = 1.0):
        """
        기존 Linear 레이어로부터 BitfieldLinear를 생성합니다.
        
        Args:
            linear: 압축할 nn.Linear 레이어
            basis_size: 기저 벡터 테이블 크기
            r_max: 최대 반지름 값
            
        Returns:
            압축된 BitfieldLinear 레이어
        """
        in_features = linear.in_features
        out_features = linear.out_features
        
        # 가중치를 numpy 배열로 변환
        weights = linear.weight.detach().cpu().numpy().astype(np.float32)
        
        # 새 인스턴스 생성
        bitfield_layer = cls(in_features, out_features, basis_size, r_max)
        
        # Rust에서 가중치 압축
        bitfield_layer.rust_layer = RustBitfieldLinear.from_weights(
            weights, basis_size, r_max
        )
        
        return bitfield_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파를 수행합니다.

        # 인자
            input: `[batch_size, in_features]` 형태의 입력 텐서.

        # 반환
            `[batch_size, out_features]` 형태의 출력 텐서.
        """
        return BitfieldLinearFunction.apply(x, self)

    def __repr__(self):
        return (f"BitfieldLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, basis_size={self.basis_size})") 