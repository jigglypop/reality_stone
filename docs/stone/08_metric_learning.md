# 08. Metric Learning — Loss, Constraints, Stability

- 파라메터화: G_k = L_k^T L_k (항상 SPD)
- 손실: Policy‑aware Contrastive (양성=접근 가능, 음성=권한 밖+마진)
- 제약: λ_min ≤ eig(G_k) ≤ λ_max, ‖log G_k‖_F 규제
- 옵티마이저: Riemannian Adam/SGD, Log‑Euclidean/Affine‑Invariant 손실
- 제품 다양체: 필요 시 혼합 곡률(Product manifold)
