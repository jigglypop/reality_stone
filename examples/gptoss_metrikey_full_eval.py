import argparse
import json
import os
import time
import numpy as np
import statistics as stats

import reality_stone as rs


def load_config_from_dir(root: str):
    cfg_path = os.path.join(root, 'config.json')
    if not os.path.isfile(cfg_path):
        # Fallback: look for config.json anywhere under root
        for dirpath, _, files in os.walk(root):
            if 'config.json' in files:
                cfg_path = os.path.join(dirpath, 'config.json')
                break
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def metrikey_metrics(d: int, L: int, precision: str = 'f64', repeat: int = 3):
    keys = [f'dept:{i}' for i in range(L)]
    masses = [1.0] * L
    lam_min, lam_max = 0.8, 1.2

    compose_fn = rs.metrikey.compose_layers_gravity_f64 if precision == 'f64' else rs.metrikey.compose_layers_gravity
    t0 = time.time()
    T = compose_fn(keys, [float(m) for m in masses] if precision == 'f64' else masses, d, float(lam_min) if precision=='f64' else lam_min, float(lam_max) if precision=='f64' else lam_max)
    compose_ms = (time.time() - t0) * 1000.0

    # apply speed
    fn_apply = rs.metrikey.apply_linear_f64 if precision == 'f64' else rs.metrikey.apply_linear
    def bench(batch):
        X = np.random.randn(batch, d).astype(np.float64 if precision == 'f64' else np.float32)
        _ = fn_apply(T, X)
        meas = []
        for _ in range(repeat):
            t1 = time.time()
            _ = fn_apply(T, X)
            meas.append((time.time() - t1) * 1000.0)
        ms = stats.median(meas)
        qps = batch / (ms / 1000.0) if ms > 0 else float('inf')
        return ms, qps

    b1 = [(b,) + bench(b) for b in (1, 64, 1024)]

    # compression ratio vs storing all T_l as f32
    orig = L * d * d * 4
    spec = L * 10 + (2 + L) * (8 if precision == 'f64' else 4) + 4
    ratio = orig / max(1, spec)

    # ranking equality (sanity)
    X = np.random.randn(512, d).astype(np.float64 if precision == 'f64' else np.float32)
    Ts = [compose_fn([keys[i]], [float(masses[i])] if precision=='f64' else [masses[i]], d, float(lam_min) if precision=='f64' else lam_min, float(lam_max) if precision=='f64' else lam_max) for i in range(L)]
    q = np.random.randn(d).astype(np.float64 if precision == 'f64' else np.float32)
    q_seq = q.copy()
    for Ti in Ts:
        q_seq = Ti @ q_seq
    q_cmp = T @ q
    d1 = np.linalg.norm(X - q_seq, axis=1)
    d2 = np.linalg.norm(X - q_cmp, axis=1)
    eq = np.array_equal(np.argsort(d1)[:10], np.argsort(d2)[:10])
    max_diff = float(np.max(np.abs(d1 - d2)))
    return compose_ms, b1, ratio, eq, max_diff


def try_generate(model_dir: str, prompt: str, max_new_tokens: int = 200):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map='auto', torch_dtype='auto')
        import torch
        input_ids = tok.encode(prompt, return_tensors='pt').to(mdl.device)
        # warmup small
        with torch.inference_mode():
            _ = mdl.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        t0 = time.time()
        with torch.inference_mode():
            out = mdl.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9, top_p=0.92, pad_token_id=tok.eos_token_id)
        lat = time.time() - t0
        gen_tokens = int(out.shape[-1] - input_ids.shape[-1])
        text = tok.decode(out[0], skip_special_tokens=True)
        return True, text, lat, gen_tokens, gen_tokens / max(1e-6, lat)
    except Exception as e:
        return False, f'[generation unavailable on this machine: {e}]', None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=r'E:/models/gpt_oss_120b')
    ap.add_argument('--precision', choices=['f32','f64'], default='f64')
    ap.add_argument('--prompt', default='''아래는 사용자와 도우미의 대화이다. 도우미는 한국어로 차분하고 상세하게 설명한다.\n\n사용자: 하이퍼볼릭 신경망과 Poincaré 모델을 간단히 설명하고, 현실 시스템(RAG)에서 왜 곡률을 조절하는 전처리가 유용한지 설명해 줘. 예시는 간단한 수식과 함께.\n도우미:''')
    args = ap.parse_args()

    cfg = load_config_from_dir(args.model_dir)
    d = int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model'))
    L = int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or cfg.get('num_layers'))
    print(f'[config] d={d}, L={L}')

    compose_ms, b1, ratio, eq, max_diff = metrikey_metrics(d, L, precision=args.precision)
    print(f'[metrikey] compose_ms={compose_ms:.2f} ms, compression_ratio={ratio:.1f}x, rank_equal={eq}, max|d1-d2|={max_diff:.3e}')
    for b, ms, qps in b1:
        print(f'[metrikey] apply: batch={b}, median_ms={ms:.2f}, qps={qps:.1f}')

    ok, text, lat, gen_tokens, tps = try_generate(args.model_dir, args.prompt, max_new_tokens=220)
    if ok:
        print('\n=== Generation (Korean) ===')
        print(text)
        print(f'\n[gen_metrics] latency_sec={lat:.3f}, gen_tokens={gen_tokens}, tokens_per_sec={tps:.2f}')
    else:
        print(text)


if __name__ == '__main__':
    main()


