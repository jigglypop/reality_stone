use ndarray::Array2;

pub type PhiField = dyn Fn(&[f64]) -> f64 + Sync + Send;

/// 리만 다양체 상의 기하학적 연산을 처리하는 트레이트
pub trait Manifold {
    /// 계량 텐서 g_uv(x) 반환
    fn metric(&self, x: &[f64]) -> Array2<f64>;

    /// 크리스토펠 기호 Gamma^k_ij(x) 반환
    fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>>;

    /// 리만 곡률 스칼라 R(x) 반환
    fn ricci_scalar(&self, x: &[f64]) -> f64;

    /// Exponential map: x_new = exp_x(v)
    /// 측지선 방정식 d^2x/dt^2 + Gamma (dx/dt)^2 = 0 을 푼다
    fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64>;
}

/// CE 이론에 따른 클라루스장 유도 계량 (Clarus Induced Metric)
/// g_uv = e^(-2 * alpha * Phi(x)) * delta_uv
/// 공간이 클라루스장 Phi에 의해 수축되는 효과를 모델링
pub struct SuppressionManifold {
    pub phi_field: Box<PhiField>,
    pub alpha: f64, // 결합 상수
    pub dim: usize,
}

impl SuppressionManifold {
    pub fn new(phi_field: Box<PhiField>, dim: usize) -> Self {
        Self {
            phi_field,
            alpha: 0.1, // 기본 결합 상수
            dim,
        }
    }

    fn gradient_phi(&self, x: &[f64]) -> Vec<f64> {
        let eps = 1e-7;
        let inv_2eps = 0.5 / eps;
        let mut grad = vec![0.0; self.dim];

        for i in 0..self.dim {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            grad[i] = ((self.phi_field)(&x_plus) - (self.phi_field)(&x_minus)) * inv_2eps;
        }
        grad
    }

    fn laplacian_phi(&self, x: &[f64]) -> f64 {
        let eps = 1e-5;
        let inv_eps2 = 1.0 / (eps * eps);
        let f0 = (self.phi_field)(x);
        let mut lap = 0.0;

        for i in 0..self.dim {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            lap += ((self.phi_field)(&xp) - 2.0 * f0 + (self.phi_field)(&xm)) * inv_eps2;
        }
        lap
    }
}

impl Manifold for SuppressionManifold {
    fn metric(&self, x: &[f64]) -> Array2<f64> {
        let phi = (self.phi_field)(x);
        let factor = (-2.0 * self.alpha * phi).exp();

        let mut g = Array2::zeros((self.dim, self.dim));
        for i in 0..self.dim {
            g[[i, i]] = factor;
        }
        g
    }

    fn christoffel(&self, x: &[f64]) -> Vec<Array2<f64>> {
        // Conformal metric g_uv = Omega^2 delta_uv, Omega = e^(-alpha*Phi)
        // Gamma^k_ij = delta^k_i d_j(ln Omega) + delta^k_j d_i(ln Omega) - delta_ij d^k(ln Omega)
        // ln Omega = -alpha * Phi

        let grad_phi = self.gradient_phi(x);
        let factor = -self.alpha;

        let mut gammas = vec![Array2::zeros((self.dim, self.dim)); self.dim];

        for k in 0..self.dim {
            for i in 0..self.dim {
                for j in 0..self.dim {
                    let term1 = if k == i { factor * grad_phi[j] } else { 0.0 };
                    let term2 = if k == j { factor * grad_phi[i] } else { 0.0 };
                    let term3 = if i == j { -factor * grad_phi[k] } else { 0.0 };

                    gammas[k][[i, j]] = term1 + term2 + term3;
                }
            }
        }
        gammas
    }

    fn ricci_scalar(&self, x: &[f64]) -> f64 {
        // Conformal metric g = e^(2f) delta, f = -alpha*Phi
        // R = -e^(-2f) [ 2(n-1) laplacian(f) + (n-2)(n-1) |grad(f)|^2 ]
        //   = e^(2*alpha*Phi) [ 2(n-1)*alpha*laplacian(Phi) - (n-2)(n-1)*alpha^2*|grad(Phi)|^2 ]

        let n = self.dim as f64;
        let phi = (self.phi_field)(x);
        let grad = self.gradient_phi(x);
        let grad_sq: f64 = grad.iter().map(|v| v * v).sum();
        let lap = self.laplacian_phi(x);

        let prefactor = (2.0 * self.alpha * phi).exp();
        let term_lap = 2.0 * (n - 1.0) * self.alpha * lap;
        let term_grad = -(n - 2.0) * (n - 1.0) * self.alpha * self.alpha * grad_sq;

        prefactor * (term_lap + term_grad)
    }

    fn exp_map(&self, x: &[f64], v: &[f64], dt: f64) -> Vec<f64> {
        // 2차 룬게-쿠타로 측지선 방정식 적분
        // d^2x^k/dt^2 = -Gamma^k_ij v^i v^j

        let gammas = self.christoffel(x);
        let mut acc = vec![0.0; self.dim];

        for k in 0..self.dim {
            let mut sum = 0.0;
            for i in 0..self.dim {
                for j in 0..self.dim {
                    sum += gammas[k][[i, j]] * v[i] * v[j];
                }
            }
            acc[k] = -sum;
        }

        let mut x_new = vec![0.0; self.dim];
        for i in 0..self.dim {
            x_new[i] = x[i] + v[i] * dt + 0.5 * acc[i] * dt * dt;
        }
        x_new
    }
}
