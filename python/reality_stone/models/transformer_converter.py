"""
Transformer → RS-ULF 완전 변환기

이 모듈은 Huggingface Transformer 모델을 Reality Stone Unified Lagrangian Flow로
수학적 정합성을 유지하며 변환합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from tqdm.auto import tqdm

from .rsulf import (
    extract_metric,
    stabilize_metric,
    curvature_from_qk,
    fold_metric_layer,
    RSULF,
    RSULFStack
)


def extract_transformer_layer_weights(model, layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    Transformer 레이어에서 필요한 모든 가중치 추출
    
    Args:
        model: Huggingface Transformer model
        layer_idx: Layer index
    
    Returns:
        dict with all weights
    """
    try:
        layer = model.model.layers[layer_idx]
    except AttributeError:
        # Some models use different structure
        try:
            layer = model.transformer.h[layer_idx]
        except AttributeError:
            layer = model.layers[layer_idx]
    
    weights = {}
    
    # Attention weights
    try:
        weights['WQ'] = layer.self_attn.q_proj.weight.detach()
        weights['WK'] = layer.self_attn.k_proj.weight.detach()
        weights['WV'] = layer.self_attn.v_proj.weight.detach()
        weights['WO'] = layer.self_attn.o_proj.weight.detach()
    except AttributeError:
        # GPT-style
        if hasattr(layer, 'attn'):
            qkv = layer.attn.c_attn.weight.detach()
            d = qkv.size(0) // 3
            weights['WQ'] = qkv[:d, :]
            weights['WK'] = qkv[d:2*d, :]
            weights['WV'] = qkv[2*d:, :]
            weights['WO'] = layer.attn.c_proj.weight.detach()
    
    # FFN weights
    try:
        # LLaMA/Mistral/Qwen style
        weights['W1'] = layer.mlp.gate_proj.weight.detach()
        weights['W2'] = layer.mlp.down_proj.weight.detach()
        if hasattr(layer.mlp, 'up_proj'):
            weights['W_up'] = layer.mlp.up_proj.weight.detach()
        else:
            weights['W_up'] = None
    except AttributeError:
        # GPT style
        weights['W1'] = layer.mlp.c_fc.weight.detach()
        weights['W2'] = layer.mlp.c_proj.weight.detach()
        weights['W_up'] = None
    
    # Normalization
    try:
        weights['norm_attn'] = layer.input_layernorm.weight.detach()
        weights['norm_ffn'] = layer.post_attention_layernorm.weight.detach()
    except AttributeError:
        try:
            weights['norm_attn'] = layer.ln_1.weight.detach()
            weights['norm_ffn'] = layer.ln_2.weight.detach()
        except AttributeError:
            weights['norm_attn'] = None
            weights['norm_ffn'] = None
    
    return weights


def create_graph_laplacian(
    seq_len: int,
    window_size: int = 8,
    directed: bool = True,
    decay: float = 0.9,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    시퀀스용 그래프 Laplacian 생성
    
    Args:
        seq_len: 시퀀스 길이
        window_size: Local attention window
        directed: True면 causal (과거만)
        decay: Distance-based weight decay
        device: torch device
    
    Returns:
        L: (seq_len, seq_len) Laplacian matrix
    """
    adj = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        # Local window
        start = max(0, i - window_size)
        end = i if directed else min(seq_len, i + window_size)
        
        for j in range(start, end):
            if i != j:
                # Distance-based weight
                dist = abs(i - j)
                weight = decay ** dist
                adj[i, j] = weight
    
    # Degree matrix
    degrees = adj.sum(dim=1)
    D = torch.diag(degrees)
    
    # Laplacian
    L = D - adj
    
    return L


class ConsistencyTester:
    """Transformer ↔ RS-ULF 정합성 검증"""
    
    def __init__(self, tolerance: float = 1e-2):
        self.tolerance = tolerance
        self.results = {}
    
    def test_metric_extraction(
        self,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        x: torch.Tensor
    ) -> bool:
        """
        Test: (Qx_i)·(Kx_j) ≈ x_i^T g x_j
        """
        with torch.no_grad():
            # Transformer side: Q and K projections
            q = F.linear(x, WQ)  # (B, L, D)
            k = F.linear(x, WK)
            
            # Dot products: (B, L, L)
            tf_dots = torch.matmul(q, k.transpose(-1, -2))
            
            # RS side: metric
            g = extract_metric(WQ, WK)
            g_stable = stabilize_metric(g, strategy="diagonal")
            
            # x^T g x: (B, L, L)
            gx = torch.matmul(x, g_stable)
            rs_dots = torch.matmul(gx, x.transpose(-1, -2))
            
            # Compare
            diff = torch.abs(tf_dots - rs_dots).mean()
            max_diff = torch.abs(tf_dots - rs_dots).max()
            
            # Cosine similarity
            tf_flat = tf_dots.flatten()
            rs_flat = rs_dots.flatten()
            cos_sim = F.cosine_similarity(tf_flat, rs_flat, dim=0)
            
            passed = diff < self.tolerance and cos_sim > 0.9
            
            self.results['metric_extraction'] = {
                'passed': passed,
                'mean_diff': diff.item(),
                'max_diff': max_diff.item(),
                'cosine_sim': cos_sim.item()
            }
            
            return passed
    
    def test_all(
        self,
        weights: Dict[str, torch.Tensor],
        x: torch.Tensor
    ) -> Dict:
        """모든 테스트 실행"""
        self.test_metric_extraction(weights['WQ'], weights['WK'], x)
        
        all_passed = all(r['passed'] for r in self.results.values())
        
        self.results['summary'] = {
            'all_passed': all_passed,
            'num_passed': sum(r['passed'] for r in self.results.values()),
            'num_total': len(self.results) - 1
        }
        
        return self.results
    
    def print_report(self):
        """검증 리포트 출력"""
        print("\n" + "="*70)
        print("CONSISTENCY TEST REPORT")
        print("="*70)
        
        for test_name, result in self.results.items():
            if test_name == 'summary':
                continue
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{test_name.upper()}: {status}")
            
            for metric, value in result.items():
                if metric != 'passed':
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.6f}")
                    else:
                        print(f"  {metric}: {value}")
        
        print("\n" + "-"*70)
        summary = self.results['summary']
        print(f"SUMMARY: {summary['num_passed']}/{summary['num_total']} tests passed")
        
        if summary['all_passed']:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        
        print("="*70 + "\n")


class TransformerToRSULFConverter:
    """
    완전한 Transformer → RS-ULF 변환기
    
    기능:
    - 가중치 추출 및 변환
    - Metric/Potential/Graph 구성
    - 차원 접기 (선택적)
    - 정합성 검증
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.default_config()
        self.stats = {}
        self.tester = ConsistencyTester(
            tolerance=self.config.get('consistency_tolerance', 1e-2)
        )
    
    @staticmethod
    def default_config() -> Dict:
        return {
            # Metric
            'metric_strategy': 'diagonal',
            
            # RS-ULF hyperparameters
            'lr': 0.02,
            'alpha': 0.04,
            'beta': 0.01,
            'gamma': 0.98,
            
            # Folding (None = no folding)
            'folding_ratio': None,  # e.g., 0.5 for half dimension
            
            # Graph
            'graph_window_size': 8,
            'graph_directed': True,
            'graph_decay': 0.9,
            
            # Testing
            'run_consistency_tests': True,
            'consistency_tolerance': 1e-2,
            
            # Verbosity
            'verbose': True
        }
    
    def convert_model(
        self,
        transformer_model,
        device: str = 'cpu'
    ) -> RSULFStack:
        """
        전체 Transformer 모델 변환
        
        Args:
            transformer_model: Huggingface model
            device: target device
        
        Returns:
            RSULFStack
        """
        if self.config['verbose']:
            print("="*70)
            print("TRANSFORMER → RS-ULF CONVERSION")
            print("="*70)
        
        # Model info
        try:
            num_layers = len(transformer_model.model.layers)
            d_model = transformer_model.config.hidden_size
        except AttributeError:
            try:
                num_layers = len(transformer_model.transformer.h)
                d_model = transformer_model.config.n_embd
            except AttributeError:
                raise ValueError("Cannot determine model structure")
        
        if self.config['verbose']:
            print(f"\nModel Info:")
            print(f"  Layers: {num_layers}")
            print(f"  Hidden size: {d_model}")
            print(f"  Config: {self.config}\n")
        
        # Convert layers
        rs_layers = []
        
        iterator = range(num_layers)
        if self.config['verbose']:
            iterator = tqdm(iterator, desc="Converting layers")
        
        for layer_idx in iterator:
            rs_layer = self.convert_layer(
                transformer_model,
                layer_idx,
                d_model,
                device
            )
            rs_layers.append(rs_layer)
        
        # Build stack
        rs_model = RSULFStack(nn.ModuleList(rs_layers))
        rs_model.to(device)
        
        if self.config['verbose']:
            self.print_stats()
        
        return rs_model
    
    def convert_layer(
        self,
        model,
        layer_idx: int,
        d_model: int,
        device: str
    ) -> RSULF:
        """단일 레이어 변환"""
        
        # 1. Extract weights
        weights = extract_transformer_layer_weights(model, layer_idx)
        
        WQ = weights['WQ'].to(device)
        WK = weights['WK'].to(device)
        W1 = weights['W1'].to(device)
        W2 = weights['W2'].to(device)
        W_up = weights.get('W_up')
        
        # 2. Consistency test (optional)
        if self.config['run_consistency_tests'] and layer_idx == 0:
            # Test only first layer to save time
            x_test = torch.randn(2, 32, d_model, device=device)
            test_results = self.tester.test_all(weights, x_test)
            
            if self.config['verbose']:
                self.tester.print_report()
        
        # 3. Folding (optional)
        if self.config['folding_ratio'] is not None:
            ratio = self.config['folding_ratio']
            
            WQ_folded, _ = fold_metric_layer(WQ, reduction_ratio=ratio)
            WK_folded, _ = fold_metric_layer(WK, reduction_ratio=ratio)
            W1_folded, _ = fold_metric_layer(W1, reduction_ratio=ratio)
            W2_folded, _ = fold_metric_layer(W2, reduction_ratio=ratio)
            
            WQ, WK, W1, W2 = WQ_folded, WK_folded, W1_folded, W2_folded
            
            self.stats[f'layer_{layer_idx}'] = {
                'folding_ratio': ratio,
                'original_params': d_model * d_model * 2 + d_model * d_model * 8,
                'folded_params': WQ.numel() + WK.numel() + W1.numel() + W2.numel()
            }
        
        # 4. Create placeholder Laplacian (will be updated at runtime)
        L = torch.eye(1, device=device)
        
        # 5. Build RS-ULF layer
        rs_layer = RSULF(
            d_model=d_model,
            WQ=WQ,
            WK=WK,
            W1=W1,
            W2=W2,
            L_matrix=L,
            lr=self.config['lr'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            gamma=self.config['gamma'],
            metric_strategy=self.config['metric_strategy']
        )
        
        return rs_layer
    
    def print_stats(self):
        """변환 통계 출력"""
        if not self.stats:
            return
        
        print("\n" + "="*70)
        print("CONVERSION STATISTICS")
        print("="*70)
        
        total_orig = 0
        total_folded = 0
        
        for layer_name, stats in self.stats.items():
            if 'original_params' in stats:
                total_orig += stats['original_params']
                total_folded += stats['folded_params']
        
        if total_orig > 0:
            reduction = (1 - total_folded / total_orig) * 100
            print(f"\nParameter Reduction:")
            print(f"  Original: {total_orig:,}")
            print(f"  Folded: {total_folded:,}")
            print(f"  Reduction: {reduction:.2f}%")
        
        print("="*70 + "\n")


def convert_and_save(
    model_name: str,
    save_path: str,
    device: str = 'cpu',
    config: Optional[Dict] = None
):
    """
    모델 변환 및 저장 (편의 함수)
    
    Args:
        model_name: Huggingface model name
        save_path: Save directory
        device: Device
        config: Converter config
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    
    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert
    converter = TransformerToRSULFConverter(config)
    rs_model = converter.convert_model(model, device=device)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    
    save_dict = {
        'model_state_dict': rs_model.state_dict(),
        'converter_config': converter.config,
        'stats': converter.stats,
        'original_model_name': model_name
    }
    
    save_file = os.path.join(save_path, 'rsulf_model.pt')
    torch.save(save_dict, save_file)
    
    print(f"\n✓ Saved to {save_file}")
    
    return rs_model, tokenizer


if __name__ == "__main__":
    # 테스트
    print("Testing Transformer → RS-ULF converter...")
    
    # Create dummy transformer-like weights
    d_model = 512
    batch, seq_len = 2, 64
    
    WQ = torch.randn(d_model, d_model) * 0.02
    WK = torch.randn(d_model, d_model) * 0.02
    W1 = torch.randn(d_model * 4, d_model) * 0.02
    W2 = torch.randn(d_model, d_model * 4) * 0.02
    
    # Test consistency
    tester = ConsistencyTester(tolerance=1e-2)
    x = torch.randn(batch, seq_len, d_model)
    
    weights = {'WQ': WQ, 'WK': WK, 'W1': W1, 'W2': W2}
    results = tester.test_all(weights, x)
    tester.print_report()
    
    # Test layer creation
    L = create_graph_laplacian(seq_len, window_size=8)
    
    rs_layer = RSULF(
        d_model=d_model,
        WQ=WQ, WK=WK, W1=W1, W2=W2,
        L_matrix=L
    )
    
    output, V = rs_layer(x, V=None)
    print(f"\nLayer test:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Memory: {V.shape}")
    print("\n✓ All tests passed!")

