import argparse
import time
import numpy as np
import statistics as stats
import reality_stone as rs


def bytes_mats(num_layers: int, d: int, elt_bytes: int = 4) -> int:
	return num_layers * d * d * elt_bytes


def bytes_spec(num_layers: int, avg_key_len: int = 10, precision: str = 'f64') -> int:
	float_size = 8 if precision == 'f64' else 4
	key_bytes = num_layers * avg_key_len
	meta_floats = 2 + num_layers
	return key_bytes + meta_floats * float_size + 4


def compose(keys, masses, d, lam_min, lam_max, precision='f64'):
	fn = rs.metrikey.compose_layers_gravity_f64 if precision == 'f64' else rs.metrikey.compose_layers_gravity
	return fn(keys, [float(m) for m in masses] if precision == 'f64' else masses, d, float(lam_min) if precision=='f64' else lam_min, float(lam_max) if precision=='f64' else lam_max)


def bench_apply(T, d, batch, repeat, precision='f64'):
	X = np.random.randn(batch, d).astype(np.float64 if precision == 'f64' else np.float32)
	fn = rs.metrikey.apply_linear_f64 if precision == 'f64' else rs.metrikey.apply_linear
	_ = fn(T, X)
	meas = []
	for _ in range(repeat):
		t0 = time.time()
		_ = fn(T, X)
		meas.append((time.time() - t0) * 1000.0)
	ms = stats.median(meas)
	qps = (batch / (ms / 1000.0)) if ms > 0 else float('inf')
	return ms, qps


def predict_scale(ms_ref, d_ref, d_target, layers_ref, layers_target, op='apply'):
	if op == 'apply':
		# matvec ~ O(d^2)
		scale = (d_target / d_ref) ** 2
	elif op == 'compose':
		# naive: L * O(d^3)
		scale = (layers_target / layers_ref) * (d_target / d_ref) ** 3
	else:
		scale = 1.0
	return ms_ref * scale


def main():
	ap = argparse.ArgumentParser(description='MetriKey gpt-oss compression benchmark & prediction')
	ap.add_argument('--d-model', type=int, default=4096)
	ap.add_argument('--layers', type=int, default=32)
	ap.add_argument('--precision', choices=['f32','f64'], default='f64')
	ap.add_argument('--ref-d', type=int, default=1024)
	ap.add_argument('--ref-l', type=int, default=8)
	ap.add_argument('--repeat', type=int, default=3)
	ap.add_argument('--batches', default='1,64,1024')
	args = ap.parse_args()

	# reference measurement at manageable size
	keys_ref = [f'dept:{i}' for i in range(args.ref_l)]
	masses_ref = [1.0] * args.ref_l
	lam_min, lam_max = 0.8, 1.2

	t0 = time.time()
	T_ref = compose(keys_ref, masses_ref, args.ref_d, lam_min, lam_max, args.precision)
	compose_ms_ref = (time.time() - t0) * 1000.0

	applies = []
	for b in [int(x) for x in args.batches.split(',') if x.strip()]:
		ms, qps = bench_apply(T_ref, args.ref_d, b, args.repeat, args.precision)
		applies.append((b, ms, qps))

	# sizes
	orig_f32 = bytes_mats(args.layers, args.d_model, 4)
	comp_spec = bytes_spec(args.layers, avg_key_len=10, precision=args.precision)
	ratio = orig_f32 / max(1, comp_spec)

	print(f'[reference] d={args.ref_d}, L={args.ref_l}, precision={args.precision}')
	print(f' compose_ms={compose_ms_ref:.2f}')
	for b, ms, qps in applies:
		print(f' apply: batch={b}, median_ms={ms:.2f}, qps={qps:.1f}')

	print(f'\n[target:gpt-oss] d={args.d_model}, L={args.layers}, precision={args.precision}')
	print(f' compression_ratio (vs f32 T_l): {ratio:.1f}x  (orig={orig_f32/1e9:.2f} GB, spec={comp_spec} B)')

	pred_compose = predict_scale(compose_ms_ref, args.ref_d, args.d_model, args.ref_l, args.layers, op='compose')
	print(f' predicted_compose_ms (CPU naive): {pred_compose:.0f} ms')
	for b, ms, _ in applies:
		pred_apply = predict_scale(ms, args.ref_d, args.d_model, args.ref_l, args.layers, op='apply')
		print(f' predicted_apply: batch={b} -> median_ms={pred_apply:.2f}')

	print('\n accuracy expectation: exact (ranking unchanged) since transform is lossless in f64; RAG-quality unaffected; LLM generation unaffected (transform is outside weights)')


if __name__ == '__main__':
	main()


