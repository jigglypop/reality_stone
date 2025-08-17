import argparse
import numpy as np
import reality_stone as rs


def make_corpus(num_docs=600, dim=256, dept_dim=32, strength=3.0):
    rng = np.random.default_rng(42)
    depts = ['AI', 'Finance', 'HR']
    per = num_docs // len(depts)
    X = rng.standard_normal((num_docs, dim)).astype(np.float64) * 0.5
    # Structured department subspace (last dept_dim)
    C = np.zeros((len(depts), dept_dim), dtype=np.float64)
    for j in range(len(depts)):
        v = rng.standard_normal(dept_dim)
        v /= np.linalg.norm(v) + 1e-9
        C[j] = v
    for j, dept in enumerate(depts):
        s = slice(j * per, (j + 1) * per)
        X[s, -dept_dim:] += strength * C[j]
    # normalize
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    payloads = [{'id': i, 'dept': depts[i % len(depts)]} for i in range(num_docs)]
    return X, payloads, depts


def key_to_L(key: str, dim: int, lam=(0.8, 1.2)):
    G = rs.metrikey.spd_metric_from_key_weighted(key, dim, lam[0], lam[1], 1.0)
    L = rs.metrikey.metric_factor_cholesky(G)
    return L.astype(np.float64)


def topk_mahalanobis(X_frame: np.ndarray, q_frame: np.ndarray, L: np.ndarray, topk: int = 10):
    # d_k(x,q) = || L (x - q) ||^2
    delta = X_frame - q_frame[None, :]
    proj = delta @ L.T
    d = np.sum(proj * proj, axis=1)
    idx = np.argsort(d)[:topk]
    return idx, d[idx]


def run_demo(dim=256, topk=10, ideal=False):
    X, payloads, depts = make_corpus(num_docs=600, dim=dim, dept_dim=32, strength=3.0)
    # Compose 3 layer keys for realism
    keys = [f"layer:{i}" for i in range(3)]
    T = rs.metrikey.compose_layers_gravity_f64(keys, [1.0, 1.0, 1.0], dim, 0.8, 1.2)

    # Authorized dept key
    key_ai = 'dept:AI'
    key_fin = 'dept:Finance'
    if ideal:
        # Idealized block scaling for crisp policy separation
        global_dim = dim - 64
        dept_dim = 64
        w_ai = 3.0
        w_fin = 1.0
        L_ai = np.eye(dim, dtype=np.float64)
        L_ai[global_dim:, global_dim:] *= w_ai
        L_fin = np.eye(dim, dtype=np.float64)
        L_fin[global_dim:, global_dim:] *= w_fin
    else:
        L_ai = key_to_L(key_ai, dim)
        L_fin = key_to_L(key_fin, dim)

    # One query aligned to AI cluster
    q = X[5].copy()
    # Apply composed frame first (single precompute)
    q_frame = (T @ q).astype(np.float64)
    X_frame = (X @ T.T).astype(np.float64)

    # Scenario 1: authorized (AI) via Mahalanobis
    idx1, _ = topk_mahalanobis(X_frame, q_frame, L_ai, topk)
    top1 = [payloads[i]['dept'] for i in idx1]

    # Scenario 2: unauthorized (Finance)
    idx2, _ = topk_mahalanobis(X_frame, q_frame, L_fin, topk)
    top2 = [payloads[i]['dept'] for i in idx2]

    # Scenario 3: grant permission (switch to AI)
    idx3, _ = topk_mahalanobis(X_frame, q_frame, L_ai, topk)
    top3 = [payloads[i]['dept'] for i in idx3]

    def pct(lst, dept):
        return 100.0 * (sum(1 for x in lst if x == dept) / len(lst))

    print('=== MetriKey RAG Auth Demo ===')
    print('dims=', dim, 'layers=3 (gravity)', 'ideal=' + str(ideal))
    print('\n[Authorized: AI] top-10 depts=', top1, f"AI%={pct(top1,'AI'):.1f}")
    print('[Unauthorized: Finance] top-10 depts=', top2, f"AI%={pct(top2,'AI'):.1f}")
    print('[Grant->AI] top-10 depts=', top3, f"AI%={pct(top3,'AI'):.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--ideal', action='store_true')
    args = ap.parse_args()
    run_demo(dim=args.dim, topk=args.topk, ideal=args.ideal)


