import torch
import pytest

from python.reality_stone.models import RCELexicalDecoder


def test_rce_decoder_shapes_and_mask_preservation():
    """
    RCELexicalDecoder가 기본적인 shape을 유지하고,
    replacement_mask=0 위치의 토큰은 항상 원본을 그대로 반환하는지 확인.
    """
    vocab_size = 100
    d_model = 32
    n_layer = 2
    n_head = 4

    model = RCELexicalDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
    )

    B, T = 2, 5
    input_ids = torch.randint(1, vocab_size, (B, T))

    # metric_ctx와 topo_idx는 현재 구현에서 사용되지 않지만,
    # API 호환성을 위해 올바른 shape으로 전달한다.
    # d_h는 여기서 d_model//n_head로 맞춰준다.
    d_h = d_model // n_head
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 3))

    # 일부 위치는 교체 불가(0), 일부는 교체 가능(1)
    replacement_mask = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1],
        ]
    )

    # 간단한 후보 사전: 각 토큰은 자기 자신과 +1 토큰만 허용
    candidates = {
        int(tid): [int(tid), min(int(tid) + 1, vocab_size - 1)]
        for tid in torch.unique(input_ids).tolist()
    }

    output_ids, logits = model(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates=candidates,
    )

    # shape 확인
    assert output_ids.shape == (B, T)
    assert logits.shape == (B, T, vocab_size)

    # replacement_mask=0인 위치는 반드시 원본 토큰 유지
    unchanged = replacement_mask == 0
    assert torch.equal(output_ids[unchanged], input_ids[unchanged])


def test_rce_decoder_respects_lexical_candidates():
    """
    후보 집합 내에서만 토큰이 선택되는지 확인.
    - replacement_mask=1 위치: output_ids는 해당 후보 집합에 포함되어야 한다.
    - replacement_mask=0 위치: 항상 원본 토큰을 유지해야 한다.
    """
    vocab_size = 50
    model = RCELexicalDecoder(vocab_size=vocab_size, d_model=32, n_layer=1, n_head=2)

    B, T = 1, 4
    # 토큰 10, 20, 30, 40
    input_ids = torch.tensor([[10, 20, 30, 40]])

    d_h = 16  # d_model//n_head 와 동일하게 맞춘다 (32//2)
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 2))

    # 두 위치는 교체, 두 위치는 고정
    replacement_mask = torch.tensor([[1, 0, 1, 0]])

    # 후보:
    #  10 -> [10, 11]
    #  20 -> [20, 21]
    #  30 -> [30] (자기 자신만)
    #  40 -> [40, 41, 42]
    candidates = {
        10: [10, 11],
        20: [20, 21],
        30: [30],
        40: [40, 41, 42],
    }

    output_ids, _ = model(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates=candidates,
    )

    # mask=0 위치는 반드시 원본 유지
    assert int(output_ids[0, 1]) == 20
    assert int(output_ids[0, 3]) == 40

    # mask=1 위치는 후보 집합 내에 있어야 한다
    assert int(output_ids[0, 0]) in candidates[10]
    assert int(output_ids[0, 2]) in candidates[30]


