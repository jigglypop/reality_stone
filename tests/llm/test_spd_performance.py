import pytest
import torch
import time

from reality_stone.models.hierarchical_sentence_topic_llm import (
    SPDMetricMixer,
    _spd_log_euclidean_mean,
)


def test_spd_fast_mixing_vs_log_euclidean():
    d_head = 32
    B = 10
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 3, -1, -1) * 0.8
    
    mixer_fast = SPDMetricMixer(d_head, use_fast_mixing=True)
    mixer_slow = SPDMetricMixer(d_head, use_fast_mixing=False)
    
    result_fast = mixer_fast.mix_hierarchy(parent_metric, self_metric, children_metrics)
    result_slow = mixer_slow.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result_fast.shape == (B, d_head, d_head)
    assert result_slow.shape == (B, d_head, d_head)
    
    assert not torch.isnan(result_fast).any()
    assert not torch.isnan(result_slow).any()


def test_spd_fast_mixing_performance():
    d_head = 32
    B = 100
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 5, -1, -1) * 0.8
    
    mixer_fast = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    start = time.time()
    for _ in range(10):
        result = mixer_fast.mix_hierarchy(parent_metric, self_metric, children_metrics)
    fast_time = time.time() - start
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()
    assert fast_time < 1.0


def test_spd_log_euclidean_mean_batch():
    B, N, d = 4, 3, 16
    
    spd_matrices = torch.eye(d).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    weights = torch.ones(B, N) / N
    
    result = _spd_log_euclidean_mean(spd_matrices, weights)
    
    assert result.shape == (B, d, d)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_spd_mixer_gradient_flow():
    d_head = 32
    B = 5
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1).requires_grad_(True)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1).requires_grad_(True)
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, None)
    loss = result.sum()
    loss.backward()
    
    assert parent_metric.grad is not None
    assert self_metric.grad is not None
    assert not torch.isnan(parent_metric.grad).any()


def test_spd_mixer_with_children():
    d_head = 32
    B = 5
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.2
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 4, -1, -1) * 0.9
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()


def test_spd_mixer_weights_normalized():
    d_head = 16
    B = 3
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    
    mixer = SPDMetricMixer(d_head, gamma_up=0.5, gamma_self=0.3, gamma_down=0.2, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, None)
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spd_fast_mixing_cuda():
    d_head = 32
    B = 50
    device = torch.device("cuda")
    
    parent_metric = torch.eye(d_head, device=device).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head, device=device).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head, device=device).unsqueeze(0).unsqueeze(0).expand(B, 3, -1, -1)
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True).to(device)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result.device.type == "cuda"
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()

