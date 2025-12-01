import torch
import pytest
import numpy as np
from reality_stone._rust import PyRSULFLayer, verify_metric_consistency


@pytest.fixture
def rs_layer():
    d = 64
    r = 16
    wq = np.random.randn(d, d).astype(np.float32) * 0.1
    wk = np.random.randn(d, d).astype(np.float32) * 0.1
    w1 = np.random.randn(d * 4, d).astype(np.float32) * 0.1
    w2 = np.random.randn(d, d * 4).astype(np.float32) * 0.1
    return PyRSULFLayer(wq, wk, w1, w2, d, r, 0.01, 0.02, 0.01, 0.99, 16, 4)


def test_메트릭_양정치성(rs_layer):
    comp = rs_layer.export_components()
    g_diag = np.array(comp['g_diag'])
    g_inv = np.array(comp['g_inv'])
    
    assert np.all(g_diag > 0)
    assert np.all(g_inv > 0)
    
    product = g_diag * g_inv
    assert np.allclose(product, 1.0, atol=1e-4)


def test_forward_출력_유한(rs_layer):
    batch = 16
    d = 64
    x = np.random.randn(batch, d).astype(np.float32)
    
    x_out, v_out = rs_layer.forward(x, None)
    
    assert x_out.shape == x.shape
    assert not np.isnan(x_out).any()
    assert not np.isinf(x_out).any()


def test_벨만_메모리_수렴():
    d = 32
    r = 8
    batch = 8
    
    wq = np.random.randn(d, d).astype(np.float32) * 0.05
    wk = np.random.randn(d, d).astype(np.float32) * 0.05
    w1 = np.random.randn(d * 4, d).astype(np.float32) * 0.05
    w2 = np.random.randn(d, d * 4).astype(np.float32) * 0.05
    
    layer = PyRSULFLayer(wq, wk, w1, w2, d, r, 0.0, 0.0, 0.0, 0.99, batch, 2)
    
    x = np.random.randn(batch, d).astype(np.float32) * 0.1
    v = None
    
    for _ in range(100):
        _, v = layer.forward(x, v)
    
    assert not np.isnan(v).any()
    assert not np.isinf(v).any()


def test_반복_안정성():
    d = 32
    r = 8
    batch = 4
    
    wq = np.random.randn(d, d).astype(np.float32) * 0.05
    wk = np.random.randn(d, d).astype(np.float32) * 0.05
    w1 = np.random.randn(d * 4, d).astype(np.float32) * 0.05
    w2 = np.random.randn(d, d * 4).astype(np.float32) * 0.05
    
    layer = PyRSULFLayer(wq, wk, w1, w2, d, r, 0.01, 0.01, 0.0, 0.99, batch, 2)
    
    x = np.random.randn(batch, d).astype(np.float32) * 0.5
    v = None
    
    for step in range(10):
        x, v = layer.forward(x, v)
        max_val = np.abs(x).max()
        assert max_val < 100.0


def test_압축률():
    d = 64
    r = 16
    
    wq = np.random.randn(d, d).astype(np.float32) * 0.1
    wk = np.random.randn(d, d).astype(np.float32) * 0.1
    w1 = np.random.randn(d * 4, d).astype(np.float32) * 0.1
    w2 = np.random.randn(d, d * 4).astype(np.float32) * 0.1
    
    layer = PyRSULFLayer(wq, wk, w1, w2, d, r, 0.01, 0.02, 0.01, 0.99, 8, 4)
    
    compressed, original, ratio = layer.param_count()
    
    assert compressed < original
    assert ratio > 1.0


def test_폴딩_정합성():
    d = 64
    r = 16
    
    wq = np.random.randn(d, d).astype(np.float32) * 0.1
    wk = np.random.randn(d, d).astype(np.float32) * 0.1
    
    result = verify_metric_consistency(wq, wk, r)
    
    assert result['fold_accuracy'] >= 0.0
    assert result['fold_accuracy'] <= 1.0


def test_곡률():
    d = 64
    r = 16
    
    wq = np.random.randn(d, d).astype(np.float32) * 0.1
    wk = np.random.randn(d, d).astype(np.float32) * 0.1
    w1 = np.random.randn(d * 4, d).astype(np.float32) * 0.1
    w2 = np.random.randn(d, d * 4).astype(np.float32) * 0.1
    
    layer = PyRSULFLayer(wq, wk, w1, w2, d, r, 0.01, 0.02, 0.01, 0.99, 8, 4)
    
    assert layer.curvature >= 0.0
