import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np

import reality_stone as rs


def bytes_for_specs(keys: List[str], masses: List[float], min_lambda: float, max_lambda: float, dim: int) -> int:
	# Rough on-wire size: UTF-8 key bytes + 2 floats (min/max) stored once + mass per layer + one int dim
	keys_bytes = sum(len(k.encode('utf-8')) for k in keys)
	floats = 2 + len(masses)  # min/max + masses
	ints = 1  # dim
	return keys_bytes + floats * 4 + ints * 4


def bytes_for_matrices(num_layers: int, dim: int) -> int:
	# Store all T_l explicitly as f32 matrices
	return num_layers * dim * dim * 4


def main():
	parser = argparse.ArgumentParser(description='MetriKey layer compression/restore demo (lossless)')
	parser.add_argument('--dim', type=int, default=256)
	parser.add_argument('--layers', type=int, default=4)
	parser.add_argument('--min-lambda', type=float, default=0.8)
	parser.add_argument('--max-lambda', type=float, default=1.2)
	parser.add_argument('--seed-prefix', dest='seed_prefix', type=str, default='dept:')
	parser.add_argument('--out', type=str, default='')
	parser.add_argument('--dataset', type=int, default=200, help='synthetic dataset size for retrieval demo')
	parser.add_argument('--k', type=int, default=5, help='top-k for retrieval demo')
	parser.add_argument('--queries', type=int, default=20, help='number of random queries for equality rate')
	parser.add_argument('--precision', choices=['f32','f64'], default='f32')
	args = parser.parse_args()

	# Construct keys and masses
	keys = [f"{args.seed_prefix}{i}" for i in range(args.layers)]
	masses = [1.0 for _ in range(args.layers)]
	d = args.dim
	lam_min, lam_max = args.min_lambda, args.max_lambda

	# Compose total transform (compressed representation is keys+masses only)
	if args.precision == 'f64':
		T_total = rs.metrikey.compose_layers_gravity_f64(keys, [float(m) for m in masses], d, float(lam_min), float(lam_max))
	else:
		T_total = rs.metrikey.compose_layers_gravity(keys, masses, d, lam_min, lam_max)

	# Recover per-layer transforms (single-element comp) and verify sequential equality
	Ts: List[np.ndarray] = []
	for i in range(args.layers):
		if args.precision == 'f64':
			Ti = rs.metrikey.compose_layers_gravity_f64([keys[i]], [float(masses[i])], d, float(lam_min), float(lam_max))
		else:
			Ti = rs.metrikey.compose_layers_gravity([keys[i]], [masses[i]], d, lam_min, lam_max)
		Ts.append(Ti)

	# Check sequential equals composed
	x = np.arange(d, dtype=(np.float64 if args.precision=='f64' else np.float32))
	x_seq = x.copy()
	for Ti in Ts:
		x_seq = Ti @ x_seq
	x_cmp = T_total @ x
	diff = float(np.linalg.norm(x_seq - x_cmp))
	print(f'composition check: ||seq - composed|| = {diff:.6e}')

	# Byte sizes
	orig_bytes = bytes_for_matrices(args.layers, d)
	comp_bytes = bytes_for_specs(keys, masses, lam_min, lam_max, d)
	ratio = orig_bytes / max(1, comp_bytes)
	print(f'size original (store all T_l): {orig_bytes/1e6:.2f} MB')
	print(f'size compressed (keys+masses): {comp_bytes} B')
	print(f'compression ratio: {ratio:.1f}x')

	# Optional payload export
	if args.out:
		spec = {
			'dim': d,
			'min_lambda': lam_min,
			'max_lambda': lam_max,
			'keys': keys,
			'masses': masses,
		}
		with open(args.out, 'w', encoding='utf-8') as f:
			json.dump(spec, f, ensure_ascii=False, indent=2)
		print(f'wrote spec to {args.out}')

	# Restore from spec and verify again
	T_rest = rs.metrikey.compose_layers_gravity(keys, masses, d, lam_min, lam_max)
	rest_diff = float(np.max(np.abs(T_rest - T_total)))
	print(f'restore check: max|T_rest - T_total| = {rest_diff:.6e}')

	# Apply to batch vectors to ensure numerically stable
	batch = 64
	X = np.random.randn(batch, d).astype(np.float64 if args.precision=='f64' else np.float32)
	if args.precision == 'f64':
		Y = rs.metrikey.apply_linear_f64(T_total, X)
	else:
		Y = rs.metrikey.apply_linear(T_total, X)
	Y_seq = X.copy()
	for Ti in Ts:
		Y_seq = (Ti @ Y_seq.T).T
	batch_diff = float(np.max(np.abs(Y - Y_seq)))
	print(f'batch application check: max|Y - Y_seq| = {batch_diff:.6e}')

	# Retrieval demo with Korean payloads: verify ranking equality (seq vs composed)
	N = int(args.dataset)
	X = np.random.randn(N, d).astype(np.float64 if args.precision=='f64' else np.float32)
	k_texts = [
		"한국어 문서 테스트", "벡터 검색 예제", "보안 정책 문서", "접근 제어 가이드",
		"사내 규정 안내", "연구 보고서", "제품 매뉴얼", "개발 문서",
	]
	payloads = [k_texts[i % len(k_texts)] for i in range(N)]

	equal_cnt = 0
	max_diff = 0.0
	for qi in range(int(args.queries)):
		q = np.random.randn(d).astype(np.float32)
		q_seq = q.copy()
		for Ti in Ts:
			q_seq = Ti @ q_seq
		q_cmp = T_total @ q

		# Euclidean retrieval on transformed query (dataset kept original)
		d1 = np.linalg.norm(X - q_seq, axis=1)
		d2 = np.linalg.norm(X - q_cmp, axis=1)
		idx1 = np.argsort(d1)[:args.k]
		idx2 = np.argsort(d2)[:args.k]
		rank_equal = np.array_equal(idx1, idx2)
		if rank_equal:
			equal_cnt += 1
		qd = float(np.max(np.abs(d1 - d2)))
		if qd > max_diff:
			max_diff = qd
		if qi < 3:
			print(f'query#{qi+1}: rank_equal={rank_equal}, max|d1-d2|={qd:.3e}')
			print(' top-k:', [payloads[i] for i in idx2])
	print(f'equality rate: {equal_cnt}/{args.queries} ({(equal_cnt/max(1,args.queries))*100:.1f}%) ; max|d1-d2|={max_diff:.3e}')


if __name__ == '__main__':
	main()
