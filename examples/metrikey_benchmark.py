import argparse
import time
import statistics as stats
import numpy as np

import reality_stone as rs


def size_bytes_spec(keys, masses, dim, precision):
	# compressed representation: keys (utf-8), min/max (float), masses (float per layer), dim (int)
	key_bytes = sum(len(k.encode('utf-8')) for k in keys)
	float_size = 8 if precision == 'f64' else 4
	meta_floats = 2 + len(masses)  # min_lambda, max_lambda, masses
	return key_bytes + meta_floats * float_size + 4  # + dim (int32)


def size_bytes_matrices(num_layers, dim, precision):
	# explicit matrices for all layers
	elt = 8 if precision == 'f64' else 4
	return num_layers * dim * dim * elt


def bench_apply(T, dim, batch, repeat, precision):
	X = np.random.randn(batch, dim).astype(np.float64 if precision == 'f64' else np.float32)
	# warmup
	_ = rs.metrikey.apply_linear_f64(T, X) if precision == 'f64' else rs.metrikey.apply_linear(T, X)
	meas = []
	for _ in range(repeat):
		t0 = time.time()
		_ = rs.metrikey.apply_linear_f64(T, X) if precision == 'f64' else rs.metrikey.apply_linear(T, X)
		meas.append((time.time() - t0) * 1000.0)
	ms = stats.median(meas)
	qps = batch / (ms / 1000.0)
	return ms, qps


def main():
	parser = argparse.ArgumentParser(description='MetriKey compressed model benchmark')
	parser.add_argument('--dim', type=int, default=1024)
	parser.add_argument('--layers', type=int, default=8)
	parser.add_argument('--min-lambda', type=float, default=0.8)
	parser.add_argument('--max-lambda', type=float, default=1.2)
	parser.add_argument('--precision', choices=['f32', 'f64'], default='f64')
	parser.add_argument('--queries', default='1,64,1024')
	parser.add_argument('--repeat', type=int, default=5)
	args = parser.parse_args()

	keys = [f'dept:{i}' for i in range(args.layers)]
	masses = [1.0] * args.layers
	d = args.dim
	lam_min, lam_max = args.min_lambda, args.max_lambda

	# Compose once from compressed spec (this is restore step)
	compose_fn = rs.metrikey.compose_layers_gravity_f64 if args.precision == 'f64' else rs.metrikey.compose_layers_gravity
	t0 = time.time()
	T = compose_fn(keys, [float(m) for m in masses] if args.precision == 'f64' else masses, d, float(lam_min) if args.precision == 'f64' else lam_min, float(lam_max) if args.precision == 'f64' else lam_max)
	compose_ms = (time.time() - t0) * 1000.0

	# Sizes and compression ratio
	orig = size_bytes_matrices(args.layers, d, 'f32')  # reference as f32
	comp = size_bytes_spec(keys, masses, d, args.precision)
	ratio = orig / max(1, comp)

	print(f'dim={d}, layers={args.layers}, precision={args.precision}')
	print(f'compose_ms={compose_ms:.2f}')
	print(f'original_f32_MB={orig/1e6:.2f}, compressed_B={comp}, compression_ratio={ratio:.1f}x')

	# Per-batch apply benchmark
	for qs in [int(x) for x in args.queries.split(',') if x.strip()]:
		ms, qps = bench_apply(T, d, qs, args.repeat, args.precision)
		print(f'apply_linear: batch={qs}, median_ms={ms:.2f}, qps={qps:.1f}')


if __name__ == '__main__':
	main()


