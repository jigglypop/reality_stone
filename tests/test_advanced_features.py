#!/usr/bin/env python3
"""
Reality Stone Advanced Features Test Suite
고급 기능들의 전체 테스트
"""

import torch
import reality_stone
import numpy as np
import time
import traceback
import pytest
import sys
import os

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

class TestAdvancedFeatures:
    """고급 기능 통합 테스트"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, method):
        """각 테스트 전에 실행"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Device: {self.device}")
        
    def test_basic_import(self):
        """기본 import 및 사용 가능한 함수 확인"""
        print_test_header("Basic Import & Function List")
        
        # 사용 가능한 함수들 출력
        funcs = [attr for attr in dir(reality_stone._C) if not attr.startswith('_')]
        print(f"✅ Reality Stone loaded successfully")
        print(f"📋 Available functions ({len(funcs)}):")
        for i, func in enumerate(sorted(funcs), 1):
            print(f"  {i:2d}. {func}")
        
        # 고급 기능 함수들이 있는지 확인
        expected_advanced = [
            'fused_linear', 'fused_mobius_chain', 'fused_transform_reg',
            'dynamic_curvature_pred', 'dynamic_mobius_add', 'dynamic_poincare_layer',
            'boundary_penalty', 'curvature_penalty', 'geodesic_penalty', 'combined_reg',
            'einstein_midpoint', 'multi_geodesic', 'geodesic_activation'
        ]
        
        missing = []
        for func in expected_advanced:
            if func not in funcs:
                missing.append(func)
        
        if missing:
            print(f"⚠️  Missing advanced functions: {missing}")
        else:
            print(f"✅ All {len(expected_advanced)} advanced functions available!")
            
        assert len(funcs) >= 42, f"Expected at least 42 functions, got {len(funcs)}"
        
    def test_fused_operations(self):
        """Fused Operations 테스트"""
        print_test_header("Fused Operations")
        
        batch_size, input_dim, output_dim = 32, 64, 128
        x = torch.randn(batch_size, input_dim, device=self.device) * 0.3
        weight = torch.randn(output_dim, input_dim, device=self.device) * 0.1
        bias = torch.randn(output_dim, device=self.device) * 0.1
        
        # 1. Fused Linear
        print("\n1️⃣ Testing fused_linear...")
        if hasattr(reality_stone._C, 'fused_linear'):
            result = reality_stone._C.fused_linear(x, weight, bias, 1.0)
            print(f"   ✅ Input: {x.shape} -> Output: {result.shape}")
            assert result.shape == (batch_size, output_dim)
            assert not torch.isnan(result).any(), "NaN values detected in fused_linear!"
        else:
            print(f"   ❌ fused_linear not available")
            pytest.skip("fused_linear not implemented")
        
        # 2. Fused Mobius Chain
        print("\n2️⃣ Testing fused_mobius_chain...")
        if hasattr(reality_stone._C, 'fused_mobius_chain'):
            inputs = [torch.randn(batch_size, input_dim, device=self.device) * 0.2 for _ in range(3)]
            curvatures = [1.0, 0.5, 2.0]
            chain_result = reality_stone._C.fused_mobius_chain(inputs, curvatures)
            print(f"   ✅ Chain of {len(inputs)} tensors -> Output: {chain_result.shape}")
            assert chain_result.shape == (batch_size, input_dim)
            assert not torch.isnan(chain_result).any(), "NaN values in chain result!"
        else:
            print(f"   ❌ fused_mobius_chain not available")
            pytest.skip("fused_mobius_chain not implemented")
        
        # 3. Transform + Regularize
        print("\n3️⃣ Testing fused_transform_reg...")
        if hasattr(reality_stone._C, 'fused_transform_reg'):
            transformed, reg_loss = reality_stone._C.fused_transform_reg(x, 1.0, 0.1)
            print(f"   ✅ Transformed: {transformed.shape}, Reg loss: {reg_loss.item():.4f}")
            assert transformed.shape == x.shape
            assert reg_loss.item() >= 0, "Regularization loss should be non-negative!"
            assert not torch.isnan(transformed).any(), "NaN in transformed output!"
        else:
            print(f"   ❌ fused_transform_reg not available")
            pytest.skip("fused_transform_reg not implemented")
        
    def test_dynamic_curvature(self):
        """Dynamic Curvature 테스트"""
        print_test_header("Dynamic Curvature")
        
        batch_size, dim = 32, 64
        
        # 1. Dynamic Curvature Prediction
        print("\n1️⃣ Testing dynamic_curvature_pred...")
        if hasattr(reality_stone._C, 'dynamic_curvature_pred'):
            features = torch.randn(batch_size, 1, device=self.device)
            weight = torch.randn(1, 1, device=self.device)
            bias = torch.randn(1, device=self.device)
            curvatures = reality_stone._C.dynamic_curvature_pred(features, weight, bias, 1.0)
            print(f"   ✅ Features: {features.shape} -> Curvatures: {curvatures.shape}")
            print(f"   📊 Curvature range: [{curvatures.min():.3f}, {curvatures.max():.3f}]")
            assert curvatures.shape == (batch_size,)
            assert (curvatures > 0).all(), "Curvatures should be positive!"
        else:
            print(f"   ❌ dynamic_curvature_pred not available")
            pytest.skip("dynamic_curvature_pred not implemented")
        
        # 2. Dynamic Mobius Add
        print("\n2️⃣ Testing dynamic_mobius_add...")
        if hasattr(reality_stone._C, 'dynamic_mobius_add'):
            u = torch.randn(batch_size, dim, device=self.device) * 0.3
            v = torch.randn(batch_size, dim, device=self.device) * 0.3
            curvatures = torch.ones(batch_size, device=self.device) * 1.0
            dynamic_result = reality_stone._C.dynamic_mobius_add(u, v, curvatures)
            print(f"   ✅ Dynamic Mobius: {u.shape} + {v.shape} -> {dynamic_result.shape}")
            assert dynamic_result.shape == u.shape
            assert not torch.isnan(dynamic_result).any(), "NaN in dynamic mobius result!"
        else:
            print(f"   ❌ dynamic_mobius_add not available")
            pytest.skip("dynamic_mobius_add not implemented")
        
        # 3. Dynamic Poincare Layer
        print("\n3️⃣ Testing dynamic_poincare_layer...")
        if hasattr(reality_stone._C, 'dynamic_poincare_layer'):
            u = torch.randn(batch_size, dim, device=self.device) * 0.3
            v = torch.randn(batch_size, dim, device=self.device) * 0.3
            curvatures = torch.ones(batch_size, device=self.device) * 1.0
            poincare_result = reality_stone._C.dynamic_poincare_layer(u, v, curvatures, 0.5)
            print(f"   ✅ Dynamic Poincare: {u.shape} interpolated with {v.shape} -> {poincare_result.shape}")
            assert poincare_result.shape == u.shape
            assert not torch.isnan(poincare_result).any(), "NaN in poincare result!"
        else:
            print(f"   ❌ dynamic_poincare_layer not available")
            pytest.skip("dynamic_poincare_layer not implemented")
        
    def test_regularization(self):
        """Regularization 테스트"""
        print_test_header("Hyperbolic Regularization")
        
        batch_size, dim = 32, 64
        x = torch.randn(batch_size, dim, device=self.device) * 0.5  # 경계 내부에
        weights = torch.randn(10, dim, device=self.device) * 0.3
        
        # 1. Boundary Penalty
        print("\n1️⃣ Testing boundary_penalty...")
        if hasattr(reality_stone._C, 'boundary_penalty'):
            boundary_loss = reality_stone._C.boundary_penalty(x, 1.0, 0.01)
            print(f"   ✅ Boundary penalty: {boundary_loss.shape}, mean: {boundary_loss.mean().item():.4f}")
            assert boundary_loss.shape == (batch_size,)
            assert (boundary_loss >= 0).all(), "Boundary penalty should be non-negative!"
        else:
            print(f"   ❌ boundary_penalty not available")
            pytest.skip("boundary_penalty not implemented")
        
        # 2. Curvature Adaptive Penalty
        print("\n2️⃣ Testing curvature_penalty...")
        if hasattr(reality_stone._C, 'curvature_penalty'):
            curvature_loss = reality_stone._C.curvature_penalty(x, 1.0)
            print(f"   ✅ Curvature penalty: {curvature_loss.shape}, mean: {curvature_loss.mean().item():.4f}")
            assert curvature_loss.shape == (batch_size,)
            assert (curvature_loss >= 0).all(), "Curvature penalty should be non-negative!"
        else:
            print(f"   ❌ curvature_penalty not available")
            pytest.skip("curvature_penalty not implemented")
        
        # 3. Geodesic Variance Penalty
        print("\n3️⃣ Testing geodesic_penalty...")
        if hasattr(reality_stone._C, 'geodesic_penalty'):
            geodesic_loss = reality_stone._C.geodesic_penalty(weights, 1.0)
            print(f"   ✅ Geodesic penalty: {geodesic_loss.shape}, value: {geodesic_loss.item():.4f}")
            assert geodesic_loss.numel() == 1, "Geodesic penalty should be scalar!"
            assert geodesic_loss.item() >= 0, "Geodesic penalty should be non-negative!"
        else:
            print(f"   ❌ geodesic_penalty not available")
            pytest.skip("geodesic_penalty not implemented")
        
        # 4. Combined Regularization
        print("\n4️⃣ Testing combined_reg...")
        if hasattr(reality_stone._C, 'combined_reg'):
            combined_loss = reality_stone._C.combined_reg(x, weights, 1.0, 0.1, 0.1, 0.1, 0.01)
            print(f"   ✅ Combined regularization: {combined_loss.shape}, value: {combined_loss.item():.4f}")
            assert combined_loss.numel() == 1, "Combined reg should be scalar!"
            assert combined_loss.item() >= 0, "Combined reg should be non-negative!"
        else:
            print(f"   ❌ combined_reg not available")
            pytest.skip("combined_reg not implemented")
        
    def test_geodesic_activation(self):
        """Geodesic Activation 테스트"""
        print_test_header("Geodesic Activation")
        
        batch_size, dim, num_anchors = 32, 64, 8
        
        input_tensor = torch.randn(batch_size, dim, device=self.device) * 0.3
        anchors = torch.randn(num_anchors, dim, device=self.device) * 0.4
        t_values = torch.randn(num_anchors, device=self.device)
        weights = torch.softmax(torch.randn(num_anchors, device=self.device), dim=0)
        
        # 1. Einstein Midpoint
        print("\n1️⃣ Testing einstein_midpoint...")
        if hasattr(reality_stone._C, 'einstein_midpoint'):
            points = torch.randn(batch_size, num_anchors, dim, device=self.device) * 0.2
            midpoint_weights = torch.softmax(torch.randn(num_anchors, device=self.device), dim=0)
            midpoint = reality_stone._C.einstein_midpoint(points, midpoint_weights, 1.0)
            print(f"   ✅ Einstein midpoint: {points.shape} -> {midpoint.shape}")
            assert midpoint.shape == (batch_size, dim)
            assert not torch.isnan(midpoint).any(), "NaN in einstein midpoint!"
        else:
            print(f"   ❌ einstein_midpoint not available")
            pytest.skip("einstein_midpoint not implemented")
        
        # 2. Multi-Geodesic Mixing
        print("\n2️⃣ Testing multi_geodesic...")
        if hasattr(reality_stone._C, 'multi_geodesic'):
            multi_result = reality_stone._C.multi_geodesic(input_tensor, anchors, t_values, weights, 1.0)
            print(f"   ✅ Multi-geodesic: {input_tensor.shape} -> {multi_result.shape}")
            assert multi_result.shape == input_tensor.shape
            assert not torch.isnan(multi_result).any(), "NaN in multi geodesic!"
        else:
            print(f"   ❌ multi_geodesic not available")
            pytest.skip("multi_geodesic not implemented")
        
        # 3. Geodesic Activation
        print("\n3️⃣ Testing geodesic_activation...")
        if hasattr(reality_stone._C, 'geodesic_activation'):
            activation_result = reality_stone._C.geodesic_activation(input_tensor, anchors, t_values, weights, 1.0)
            print(f"   ✅ Geodesic activation: {input_tensor.shape} -> {activation_result.shape}")
            assert activation_result.shape == input_tensor.shape
            assert not torch.isnan(activation_result).any(), "NaN in geodesic activation!"
        else:
            print(f"   ❌ geodesic_activation not available")
            pytest.skip("geodesic_activation not implemented")
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_performance(self):
        """성능 테스트"""
        print_test_header("Performance Test")
        
        batch_size, dim = 1024, 256
        num_iterations = 100
        
        x = torch.randn(batch_size, dim, device='cuda')
        y = torch.randn(batch_size, dim, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = reality_stone._C.mobius_add_cuda(x, y, 1.0)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = reality_stone._C.mobius_add_cuda(x, y, 1.0)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = (batch_size * num_iterations) / (end_time - start_time)
        
        print(f"   ✅ Mobius Add CUDA Performance:")
        print(f"      🚀 Average time: {avg_time:.3f} ms")
        print(f"      📈 Throughput: {throughput:.0f} samples/sec")
        
        assert avg_time < 1.0, f"Performance regression: {avg_time:.3f}ms > 1.0ms"
        
    def test_numerical_stability(self):
        """수치적 안정성 테스트 (NaN 문제 해결 확인)"""
        print_test_header("Numerical Stability Test")
        
        # 경계에 가까운 값들로 테스트
        edge_values = torch.tensor([
            [0.99, 0.0],   # 거의 경계
            [0.0, 0.99],   
            [-0.99, 0.0],  
            [0.0, -0.99],  
            [0.98, 0.02],  # 경계 근처 조합
        ], device=self.device)
        
        print(f"\n🔍 Testing edge cases near boundary...")
        
        # 기본 Mobius 연산 테스트
        for i, val in enumerate(edge_values):
            for j in range(i+1, len(edge_values)):
                result = reality_stone._C.mobius_add_cuda(val.unsqueeze(0), edge_values[j].unsqueeze(0), 1.0)
                
                # NaN 체크
                if torch.isnan(result).any():
                    print(f"   ❌ NaN detected: {val} + {edge_values[j]} = {result}")
                    assert False, f"NaN values at edge case {i},{j}"
                
                # 경계 체크
                norm = torch.norm(result, dim=-1)
                if (norm >= 1.0).any():
                    print(f"   ⚠️  Boundary violation: ||{result}|| = {norm}")
        
        print(f"   ✅ All edge cases passed numerical stability test!")


if __name__ == "__main__":
    # pytest 없이 직접 실행할 때
    print("🌟 Reality Stone Advanced Features Test Suite")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    tester = TestAdvancedFeatures()
    tester.setup_method(None)
    
    test_methods = [
        tester.test_basic_import,
        tester.test_fused_operations,
        tester.test_dynamic_curvature,
        tester.test_regularization,
        tester.test_geodesic_activation,
        tester.test_numerical_stability,
    ]
    
    if torch.cuda.is_available():
        test_methods.append(tester.test_performance)
    
    passed = 0
    total = len(test_methods)
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
            print(f"✅ {test_method.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_method.__name__} FAILED: {e}")
            traceback.print_exc()
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Reality Stone advanced features are working!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check implementation.")
        sys.exit(1) 