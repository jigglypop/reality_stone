import torch
import torch.nn.functional as F
import pytest
from transformers import GPT2LMHeadModel
from reality_stone.models.transformer_converter import RSULFTransformerConverter
from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPT-2 RS-ULF last layer test")
def test_gpt2_last_layer_rsulf_forward():
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    converter = RSULFTransformerConverter(
        d_model=model.config.n_embd,
        r=model.config.n_embd,
        eta=0.005,
        alpha=0.01,
        beta=0.0,
        gamma=0.99,
        seq_len=16,
        window=4,
        verbose=False,
        exact=True,
    )
    blocks = list(model.transformer.h)
    last_idx = len(blocks) - 1
    last_block = blocks[last_idx]
    weights = converter.extract_weights(last_block)
    rsulf = RSULFLayerCUDA(
        wq=weights["WQ"],
        wk=weights["WK"],
        w1=weights["W1"],
        w2=weights["W2"],
        d_model=model.config.n_embd,
        r=model.config.n_embd,
        eta=0.005,
        alpha=0.01,
        beta=0.0,
        gamma=0.99,
        seq_len=16,
        window=4,
        global_basis=None,
    )
    if "ln_1_weight" in weights:
        rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
    if "ln_2_weight" in weights:
        rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
    rsulf.to(device)
    batch = 2
    seq_len = 16
    d_model = model.config.n_embd
    num_samples = 8
    cos_list = []
    rel_list = []
    for step in range(num_samples):
        x = torch.randn(batch, seq_len, d_model, device=device)
        with torch.no_grad():
            out_teacher = last_block(x)
            teacher_out = out_teacher[0] if isinstance(out_teacher, (tuple, list)) else out_teacher
            student_out, _ = rsulf(x)
        assert teacher_out.shape == student_out.shape
        assert torch.isfinite(student_out).all()
        t_flat = teacher_out.view(-1, d_model)
        s_flat = student_out.view(-1, d_model)
        cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
        rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
        cos_list.append(cos)
        rel_list.append(rel.item())
        print(f"[gpt2_last_layer] sample={step+1}/{num_samples} cos={cos:.4f}, rel_l2={rel:.4f}")
    mean_cos = sum(cos_list) / len(cos_list)
    mean_rel = sum(rel_list) / len(rel_list)
    print(f"[gpt2_last_layer] mean cos={mean_cos:.4f}, mean_rel_l2={mean_rel:.4f}")
