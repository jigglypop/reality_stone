import pytest
import torch
import torch.nn as nn
from reality_stone import GeodesicMLP

class TestModels:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def sample_data(self, device):
        batch_size = 32
        input_dim = 784
        output_dim = 10
        x = torch.randn(batch_size, input_dim).to(device)
        y = torch.randint(0, output_dim, (batch_size,)).to(device)
        return x, y
    
    def test_model_output_shape(self, device, sample_data):
        x, _ = sample_data
        model = GeodesicMLP(in_dim=784, hid=128, out_dim=10).to(device)
        output = model(x)
        assert output.shape == (x.size(0), 10)
    
    def test_model_training_step(self, device, sample_data):
        x, y = sample_data
        model = GeodesicMLP(in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        output = model(x)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        output = model(x)
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_model_nan_handling(self, device):
        """NaN 처리 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5, c=1e-3, t=0.5).to(device)
        extreme_inputs = [
            torch.ones(5, 10).to(device) * 1e6,  # 매우 큰 값
            torch.ones(5, 10).to(device) * 1e-6,  # 매우 작은 값
            torch.zeros(5, 10).to(device),  # 모두 0
        ]
        for x in extreme_inputs:
            output = model(x)
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
    
    @pytest.mark.parametrize("t_value", [0.1, 0.5, 1.0, 10.0])
    def test_geodesic_mlp_t_parameter(self, device, t_value):
        """GeodesicMLP의 t 패러미터 영향 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5, t=t_value).to(device)
        x = torch.randn(8, 10).to(device)
        
        output = model(x)
        assert output.shape == (8, 5)
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("c_value", [1e-4, 1e-3, 1e-2, 0.1])
    def test_geodesic_mlp_curvature(self, device, c_value):
        """GeodesicMLP의 곡률 패러미터 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5, c=c_value).to(device)
        x = torch.randn(8, 10).to(device)
        
        output = model(x)
        assert output.shape == (8, 5)
        assert not torch.isnan(output).any()
    
    def test_model_save_load(self, device, tmp_path):
        """모델 저장/로딩 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5).to(device)
        x = torch.randn(4, 10).to(device)
        
        # 원래 출력
        original_output = model(x)
        
        # 저장
        save_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), save_path)
        
        # 새 모델에 로딩
        new_model = GeodesicMLP(in_dim=10, hid=20, out_dim=5).to(device)
        new_model.load_state_dict(torch.load(save_path))
        
        # 출력 비교
        new_output = new_model(x)
        assert torch.allclose(original_output, new_output)
    
    def test_batch_consistency(self, device):
        """배치 처리 일관성 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5).to(device)
        model.eval()
        
        # 단일 샘플
        x_single = torch.randn(1, 10).to(device)
        out_single = model(x_single)
        
        # 배치에서 같은 샘플
        x_batch = x_single.repeat(5, 1)
        out_batch = model(x_batch)
        
        # 모든 출력이 같아야 함
        for i in range(5):
            assert torch.allclose(out_single[0], out_batch[i])
    
    def test_geodesic_stability_over_time(self, device):
        """시간에 따른 측지선 안정성 테스트"""
        model = GeodesicMLP(in_dim=10, hid=20, out_dim=5, t=0.5).to(device)
        x = torch.randn(4, 10).to(device)
        
        # 모델을 여러 번 실행
        outputs = []
        for _ in range(5):
            output = model(x)
            outputs.append(output)
        
        # 모든 실행이 같은 결과를 내야 함
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])
    
    def test_model_memory_efficiency(self, device):
        """메모리 효율성 테스트"""
        if device.type != "cuda":
            pytest.skip("CUDA only test")
        
        model = GeodesicMLP(in_dim=1000, hid=500, out_dim=100).to(device)
        x = torch.randn(64, 1000).to(device)
        
        # 메모리 사용량 측정
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 전방향 패스
        output = model(x)
        forward_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 역전파
        loss = output.sum()
        loss.backward()
        total_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Forward memory: {forward_memory:.2f} MB")
        print(f"Total memory: {total_memory:.2f} MB")
        
        # 합리적인 메모리 사용량인지 확인
        assert total_memory < 2000  # 2GB 미만
    
    def test_different_hidden_dimensions(self, device):
        """다양한 은닉층 차원에서의 테스트"""
        hidden_dims = [16, 32, 64, 128, 256]
        
        for hid in hidden_dims:
            model = GeodesicMLP(in_dim=50, hid=hid, out_dim=10).to(device)
            x = torch.randn(8, 50).to(device)
            
            output = model(x)
            assert output.shape == (8, 10)
            assert not torch.isnan(output).any()
    
    def test_extreme_t_values(self, device):
        extreme_t_values = [0.0, 1e-6, 100.0, 1000.0]
        for t in extreme_t_values:
            model = GeodesicMLP(in_dim=10, hid=20, out_dim=5, t=t).to(device)
            x = torch.randn(4, 10).to(device)
            output = model(x)
            assert not torch.isnan(output).any(), f"NaN with t={t}"
            assert not torch.isinf(output).any(), f"Inf with t={t}"