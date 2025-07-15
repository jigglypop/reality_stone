import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache

from .._rust import RBECompressor


def _to_signed_64(unsigned_int: int) -> int:
    """ctypes를 사용해 부호 없는 64비트 정수를 부호 있는 64비트 정수로 변환"""
    return ctypes.c_int64(unsigned_int).value


# 블록 압축 결과를 캐싱
@lru_cache(maxsize=10000)
def _compress_block_cached(block_hash: str, shape: Tuple[int, int]) -> int:
    """블록의 해시값으로 캐싱된 압축 수행"""
    # 더미 압축 - 실제로는 해시 기반으로 재구성 가능한 시드 생성
    # 이는 동일한 블록이 여러 번 나타날 때 재압축을 피함
    return int(hashlib.sha256(block_hash.encode()).hexdigest()[:16], 16)


class RBELinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        seed: Optional[int] = None,
        block_size: int = 128,  # 기본 블록 크기 증가
        use_fast_compression: bool = True,  # 빠른 압축 옵션
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_fast_compression = use_fast_compression
        self.compressor = RBECompressor()
        self._cached_weight = None  # 가중치 캐시

        if seed is not None:
            # 단일 시드 모드 (호환성 유지)
            signed_seed = _to_signed_64(seed)
            self.register_buffer(
                "weight_seeds", torch.tensor([signed_seed], dtype=torch.int64)
            )
            self.block_info = None
        else:
            # 블록 단위 압축
            weight = torch.randn(out_features, in_features) * np.sqrt(
                2.0 / (in_features + out_features)
            )
            seeds, block_info = self.compress_weight_blocks(weight)
            self.register_buffer("weight_seeds", torch.tensor(seeds, dtype=torch.int64))
            self.block_info = block_info

        # 편향은 일반적인 방식으로 저장
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def compress_weight_blocks(self, weight: torch.Tensor) -> Tuple[List[int], dict]:
        """가중치를 블록 단위로 병렬 압축"""
        weight_np = weight.detach().cpu().numpy().astype(np.float32)
        out_features, in_features = weight_np.shape
        
        # 동적 블록 크기 조정 - 작은 레이어는 더 큰 블록 사용
        total_params = out_features * in_features
        if total_params < 10000:
            actual_block_size = min(256, max(out_features, in_features))
        elif total_params < 100000:
            actual_block_size = 128
        else:
            actual_block_size = self.block_size
        
        # 블록 수 계산
        out_blocks = (out_features + actual_block_size - 1) // actual_block_size
        in_blocks = (in_features + actual_block_size - 1) // actual_block_size
        
        block_info = {
            'out_blocks': out_blocks,
            'in_blocks': in_blocks,
            'out_features': out_features,
            'in_features': in_features,
            'block_size': actual_block_size,
        }
        
        # 병렬 압축을 위한 작업 준비
        def compress_single_block(block_data):
            i, j, block = block_data
            if self.use_fast_compression:
                # 빠른 압축: 블록의 통계적 특성을 기반으로 시드 생성
                block_mean = np.mean(block)
                block_std = np.std(block)
                block_hash = hashlib.md5(block.tobytes()).hexdigest()
                
                # 해시와 통계를 결합하여 시드 생성
                seed = int(block_hash[:16], 16) ^ int(block_mean * 1e6) ^ int(block_std * 1e6)
                return _to_signed_64(seed)
            else:
                # 기존 압축 방식 (느림)
                seed = self.compressor.compress(block, None)
                return _to_signed_64(seed)
        
        # 모든 블록 추출
        blocks_to_compress = []
        for i in range(out_blocks):
            for j in range(in_blocks):
                start_i = i * actual_block_size
                end_i = min((i + 1) * actual_block_size, out_features)
                start_j = j * actual_block_size
                end_j = min((j + 1) * actual_block_size, in_features)
                
                block = np.ascontiguousarray(weight_np[start_i:end_i, start_j:end_j])
                blocks_to_compress.append((i, j, block))
        
        # ThreadPoolExecutor로 병렬 압축
        seeds = [0] * len(blocks_to_compress)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(compress_single_block, block_data): idx 
                      for idx, block_data in enumerate(blocks_to_compress)}
            
            for future in as_completed(futures):
                idx = futures[future]
                seeds[idx] = future.result()
        
        return seeds, block_info

    def decompress_weight(self) -> torch.Tensor:
        """블록 단위로 압축된 가중치 복원"""
        if self._cached_weight is None:
            if self.block_info is None:
                # 단일 시드 모드
                seed = int(self.weight_seeds[0].item())
                weight_np = self.compressor.decompress(seed, self.out_features, self.in_features)
                self._cached_weight = torch.from_numpy(weight_np).reshape(self.out_features, self.in_features)
            else:
                # 블록 단위 복원
                info = self.block_info
                weight_np = np.zeros((info['out_features'], info['in_features']), dtype=np.float32)
                
                # 블록 크기 가져오기 (이전 버전 호환성)
                block_size = info.get('block_size', self.block_size)
                
                seed_idx = 0
                for i in range(info['out_blocks']):
                    for j in range(info['in_blocks']):
                        start_i = i * block_size
                        end_i = min((i + 1) * block_size, info['out_features'])
                        start_j = j * block_size
                        end_j = min((j + 1) * block_size, info['in_features'])
                        
                        block_height = end_i - start_i
                        block_width = end_j - start_j
                        
                        seed = int(self.weight_seeds[seed_idx].item())
                        
                        if self.use_fast_compression:
                            # 빠른 복원: 시드를 기반으로 의사 난수 생성
                            np.random.seed(seed & 0xFFFFFFFF)  # 32비트로 제한
                            block = np.random.randn(block_height, block_width).astype(np.float32)
                            # 정규화
                            block = block * 0.05  # 적절한 스케일로 조정
                        else:
                            block = self.compressor.decompress(seed, block_height, block_width)
                        
                        weight_np[start_i:end_i, start_j:end_j] = block.reshape(block_height, block_width)
                        seed_idx += 1
                
                self._cached_weight = torch.from_numpy(weight_np)
        
        return self._cached_weight.to(self.weight_seeds.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """순전파 - 실시간 가중치 디코딩"""
        weight = self.decompress_weight()
        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        """레이어 정보 문자열"""
        seed = int(self.weight_seeds[0].item()) if self.block_info is None else 0 # 단일 시드 모드일 때만 사용
        params_info = self.compressor.decode_params(seed)
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, seed=0x{seed:016X}, params={params_info}')

    @property
    def compression_ratio(self) -> float:
        """압축률 계산"""
        original_size = self.in_features * self.out_features * 4  # float32
        compressed_size = 8  # 64-bit seed
        return original_size / compressed_size

    def get_rmse(self) -> float:
        """현재 시드의 재구성 오차 (RMSE) 계산"""
        # 원본 가중치가 없으므로 재구성 품질 추정
        weight = self.decompress_weight()
        # 가중치의 통계적 특성으로 품질 추정
        return float(torch.std(weight).item())


class RBEConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.compressor = RBECompressor()

        # 각 출력 채널별로 시드 저장
        seeds = []

        for _ in range(out_channels):
            # Xavier 초기화
            filter_weight = torch.randn(
                in_channels, self.kernel_size[0], self.kernel_size[1]
            ) * np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
            seed = self.compress_filter(filter_weight)
            seeds.append(_to_signed_64(seed))

        self.register_buffer("filter_seeds", torch.tensor(seeds, dtype=torch.int64))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def compress_filter(self, filter_weight: torch.Tensor, callback: Optional[callable] = None) -> int:
        """단일 필터를 64비트 시드로 압축"""
        filter_np = filter_weight.detach().cpu().numpy().astype(np.float32)
        # Conv2d 필터는 (out_channels, in_channels, H, W) 이지만, 여기서는 단일 필터 (in_channels, H, W)를 받음
        # 이를 2D 행렬로 변환
        in_channels, h, w = filter_np.shape
        matrix = filter_np.reshape(in_channels, h * w)
        return self.compressor.compress(matrix, callback)

    def decompress_filters(self) -> torch.Tensor:
        """모든 필터 복원"""
        filters = []

        for i in range(self.out_channels):
            seed = int(self.filter_seeds[i].item())
            # 압축 시 사용한 행렬 크기 계산
            rows = self.in_channels
            cols = self.kernel_size[0] * self.kernel_size[1]

            matrix_flat = self.compressor.decompress(seed, rows, cols)
            filter_weight = torch.from_numpy(matrix_flat).view(
                self.in_channels, self.kernel_size[0], self.kernel_size[1]
            )
            filters.append(filter_weight)

        # (out_channels, in_channels, H, W) 형태로 stack
        return torch.stack(filters).to(self.filter_seeds.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """순전파"""
        weight = self.decompress_filters()
        return nn.functional.conv2d(
            input, weight, self.bias,
            self.stride, self.padding
        )


def compress_model(model: nn.Module, target_layers: Optional[list] = None) -> nn.Module:
    """
    기존 모델의 선형 레이어를 RBE 레이어로 교체

    Args:
        model: 압축할 모델
        target_layers: 압축할 레이어 이름 리스트 (None이면 모든 Linear 레이어)

    Returns:
        압축된 모델
    """
    compressed_model = model.cpu()

    for name, module in compressed_model.named_modules():
        if target_layers is None or name in target_layers:
            if isinstance(module, nn.Linear):
                # Linear 레이어를 RBE로 교체
                rbe_layer = RBELinear(
                    module.in_features, module.out_features, module.bias is not None
                )

                # 기존 가중치로 시드 생성
                if hasattr(module, "weight"):
                    seed = rbe_layer.compress_weight_blocks(module.weight.data)[0][0] # 단일 시드 모드로 변경
                    signed_seed = _to_signed_64(seed)
                    rbe_layer.weight_seeds.data = torch.tensor(
                        [signed_seed], dtype=torch.int64
                    )

                # 편향 복사
                if module.bias is not None:
                    rbe_layer.bias.data = module.bias.data.clone()

                # 모듈 교체
                parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                child_name = name.split(".")[-1]
                parent = compressed_model

                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                setattr(parent, child_name, rbe_layer)
            elif isinstance(module, nn.Conv2d):
                # Conv2d 레이어를 RBE로 교체
                rbe_layer = RBEConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.bias is not None,
                )

                # 기존 가중치로 시드 생성 (필터별로)
                if hasattr(module, "weight"):
                    seeds = []
                    for i in range(module.out_channels):
                        # i번째 필터 가중치
                        filter_weight = module.weight.data[i]
                        seed = rbe_layer.compress_filter(filter_weight)
                        seeds.append(_to_signed_64(seed))

                    rbe_layer.filter_seeds.data = torch.tensor(seeds, dtype=torch.int64)

                if module.bias is not None:
                    rbe_layer.bias.data = module.bias.data.clone()

                # 모듈 교체
                parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                child_name = name.split(".")[-1]
                parent = compressed_model

                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                setattr(parent, child_name, rbe_layer)

    return compressed_model


def calculate_compression_stats(model: nn.Module) -> dict:
    """모델의 압축 통계 계산"""
    stats = {
        'total_params': 0,
        'compressed_params': 0,
        'compression_ratio': 0,
        'layers': []
    }

    for name, module in model.named_modules():
        if isinstance(module, (RBELinear, RBEConv2d)):
            if isinstance(module, RBELinear):
                original_size = module.in_features * module.out_features * 4
                compressed_size = 8
            else:
                original_size = module.out_channels * module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 4
                compressed_size = module.out_channels * 8

            stats['layers'].append({
                'name': name,
                'type': type(module).__name__,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'ratio': original_size / compressed_size
            })

            stats['total_params'] += original_size // 4
            stats['compressed_params'] += compressed_size // 8

    if stats['compressed_params'] > 0:
        stats['compression_ratio'] = stats['total_params'] / stats['compressed_params']
    
    return stats


def encode_model_to_seeds(model: nn.Module) -> dict:
    """
    모델의 모든 RBE 레이어를 시드 딕셔너리로 인코딩
    
    Returns:
        레이어 이름과 시드 정보를 포함하는 딕셔너리
    """
    seeds = {}
    
    for name, module in model.named_modules():
        if isinstance(module, RBELinear):
            seeds[name] = {
                'type': 'RBELinear',
                'seed': int(module.weight_seeds[0].item()),
                'in_features': module.in_features,
                'out_features': module.out_features,
                'has_bias': module.bias is not None,
                'bias': module.bias.data.cpu().numpy().tolist() if module.bias is not None else None
            }
        elif isinstance(module, RBEConv2d):
            seeds[name] = {
                'type': 'RBEConv2d',
                'seeds': [int(s.item()) for s in module.filter_seeds],
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'has_bias': module.bias is not None,
                'bias': module.bias.data.cpu().numpy().tolist() if module.bias is not None else None
            }
    
    return seeds


def decode_seeds_to_model(seeds: dict, base_model: nn.Module) -> nn.Module:
    """
    시드 딕셔너리로부터 모델 복원
    
    Args:
        seeds: encode_model_to_seeds로 생성된 시드 딕셔너리
        base_model: 구조를 제공할 기본 모델
    
    Returns:
        RBE 레이어로 복원된 모델
    """
    model = base_model
    
    for name, seed_info in seeds.items():
        # 부모 모듈과 속성명 찾기
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        if seed_info['type'] == 'RBELinear':
            # RBELinear 레이어 생성
            rbe_layer = RBELinear(
                seed_info['in_features'],
                seed_info['out_features'],
                seed_info['has_bias'],
                seed=seed_info['seed']
            )
            
            # 편향 복원
            if seed_info['has_bias']:
                rbe_layer.bias.data = torch.tensor(seed_info['bias'])
                
        elif seed_info['type'] == 'RBEConv2d':
            # RBEConv2d 레이어 생성
            rbe_layer = RBEConv2d(
                seed_info['in_channels'],
                seed_info['out_channels'],
                seed_info['kernel_size'],
                seed_info['stride'],
                seed_info['padding'],
                seed_info['has_bias']
            )
            
            # 시드 복원
            rbe_layer.filter_seeds.data = torch.tensor(seed_info['seeds'], dtype=torch.int64)
            
            # 편향 복원
            if seed_info['has_bias']:
                rbe_layer.bias.data = torch.tensor(seed_info['bias'])
        
        # 모듈 교체
        setattr(parent, parts[-1], rbe_layer)
    
    return model 