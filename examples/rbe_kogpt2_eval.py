import argparse
import math
import time
from typing import List, Tuple

import numpy as np
import torch

try:
    import reality_stone as rs
    RBECompressor = rs._rust.RBECompressor
except Exception as e:  # Fallback if package layout differs
    from reality_stone._rust import RBECompressor  # type: ignore


def u64_to_i64(u: int) -> int:
    return int(u) if u <= 0x7FFFFFFFFFFFFFFF else int(u - (1 << 64))


def list_target_mats(model) -> List[Tuple[str, torch.nn.Parameter]]:
    names: List[Tuple[str, torch.nn.Parameter]] = []
    for n, p in model.named_parameters():
        if p.ndim == 2 and ('wte' not in n) and ('lm_head' not in n):
            if any(k in n for k in [
                'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'
            ]):
                names.append((n, p))
    return names


def main():
    parser = argparse.ArgumentParser(description='RBE compress/restore KoGPT2 layers and verify')
    parser.add_argument('--model', default='skt/kogpt2-base-v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-layers', type=int, default=8, help='number of 2D weight matrices to process')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--prompt', default='질문: 대한민국의 수도는?\n답변:')
    parser.add_argument('--no-generate', action='store_true')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)

    print(f'Load model: {args.model}')
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()

    mats = list_target_mats(model)
    print(f'Target matrices found: {len(mats)} (processing up to {args.max_layers})')

    comp = RBECompressor()

    orig_bytes_total = 0
    comp_bytes_total = 0
    processed = 0
    layer_stats: List[Tuple[str, Tuple[int, int], float]] = []
    t0 = time.time()

    for name, p in mats:
        if processed >= args.max_layers:
            break
        w = p.detach().to('cpu').float().numpy()
        rows, cols = w.shape
        seed_u = comp.compress(w)
        seed_s = u64_to_i64(seed_u)
        rec = np.array(comp.decompress(seed_s, rows, cols), dtype=np.float32).reshape(rows, cols)
        rmse = float(np.sqrt(np.mean((w - rec) ** 2)))
        p.data.copy_(torch.from_numpy(rec).to(p.device))

        ob = rows * cols * 4
        cb = 8
        orig_bytes_total += ob
        comp_bytes_total += cb
        processed += 1
        layer_stats.append((name, (rows, cols), rmse))
        print(f'[{processed}/{args.max_layers}] {name}: shape=({rows},{cols}), rmse={rmse:.4f}, orig={ob/1e6:.2f}MB -> comp=8B')

    elapsed = time.time() - t0
    ratio = (orig_bytes_total / comp_bytes_total) if comp_bytes_total > 0 else float('inf')
    print(f'\nSummary: layers={processed}, time={elapsed:.1f}s, total_orig={orig_bytes_total/1e6:.2f}MB, total_comp={comp_bytes_total}B, ratio={ratio:.1f}x')
    if layer_stats:
        rmses = [rm for _, _, rm in layer_stats]
        print(f'RMSE stats: min={min(rmses):.4f}, med={np.median(rmses):.4f}, max={max(rmses):.4f}')

    if not args.no-generate:
        print('\n=== Generation check ===')
        input_ids = tok.encode(args.prompt, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(text)


if __name__ == '__main__':
    main()


