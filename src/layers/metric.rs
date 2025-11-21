// ============================================================================
// 파일: src/layers/metric.rs
// 목적: 리만 메트릭 텐서의 추상화 및 구현
// ============================================================================

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

const EPS: f32 = 1e-7;

/// 리만 메트릭 텐서의 공통 인터페이스
pub trait MetricTensor: Send + Sync {
    /// 메트릭 텐서 g_ij(x) 계산 (대각 원소만 반환, batch x dim)
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;
    
    /// 역메트릭 g^ij(x) 계산 (대각 원소만)
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32>;
    
    /// 크리스토펠 기호 Γ^k_ij 계산 (대각 근사, batch별)
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>>;
    
    /// 리만 거리 d_g(x, y)
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32>;
    
    /// 메트릭의 행렬식 det(g)
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32>;
    
    /// 곡률 스칼라
    fn curvature(&self) -> f32;
}

/// 대각 메트릭 (구현 효율성을 위한 근사)
/// g_ij(x) = w_i(x) δ_ij
pub struct DiagonalMetric {
    /// 각 차원의 가중치를 계산하는 함수 파라미터
    pub weights: Array1<f32>,  // learnable parameters
    pub base_weight: f32,
}

impl DiagonalMetric {
    pub fn new(dim: usize) -> Self {
        Self {
            weights: Array1::ones(dim),
            base_weight: 1.0,
        }
    }
    
    /// w_i(x) = softplus(weights[i] * x[i]) + ε
    fn compute_weights(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(x.raw_dim());
        for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                let z = self.weights[j] * x[[i, j]];
                *val = softplus(z) + EPS;
            }
        }
        result
    }
}

impl MetricTensor for DiagonalMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        self.compute_weights(x)
    }
    
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let weights = self.compute_weights(x);
        weights.mapv(|w| 1.0 / w.max(EPS))
    }
    
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // 대각 메트릭에서 Γ^i_ii = (1/2w_i) * dw_i/dx_i
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut symbols = Vec::new();
        
        for i in 0..batch_size {
            let mut gamma = Array2::zeros((dim, dim));
            for j in 0..dim {
                let x_val = x[[i, j]];
                let w = softplus(self.weights[j] * x_val) + EPS;
                // d(softplus(z))/dz = sigmoid(z)
                let dw_dx = self.weights[j] * sigmoid(self.weights[j] * x_val);
                gamma[[j, j]] = 0.5 * dw_dx / w;
            }
            symbols.push(gamma);
        }
        symbols
    }
    
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        // 유클리드 거리의 메트릭 가중 버전
        let diff = x - y;
        let weights = self.compute_weights(x);
        let weighted_sq = &diff * &diff * &weights;
        weighted_sq.sum_axis(Axis(1)).mapv(|s| s.sqrt())
    }
    
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let weights = self.compute_weights(x);
        weights.axis_iter(ndarray::Axis(1))
            .map(|row| row.iter().product())
            .collect()
    }
    
    fn curvature(&self) -> f32 {
        0.0  // 대각 메트릭의 곡률은 0 (국소적으로 평탄)
    }
}

/// 푸앵카레 메트릭
/// g_ij(x) = (2/(1-c||x||²))² δ_ij
pub struct PoincareMetric {
    pub curvature: f32,
}

impl PoincareMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }
    
    fn conformal_factor(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let denom = (1.0 - self.curvature * &x_norm_sq).mapv(|v| v.max(EPS));
        (2.0 / denom).mapv(|v| v * v)
    }
}

impl MetricTensor for PoincareMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let lambda_sq = self.conformal_factor(x);
        // g_ij = λ² δ_ij, 대각만 저장
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));
        for i in 0..batch_size {
            for j in 0..dim {
                metric[[i, j]] = lambda_sq[i];
            }
        }
        metric
    }
    
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let lambda_sq = self.conformal_factor(x);
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut inv_metric = Array2::zeros((batch_size, dim));
        for i in 0..batch_size {
            for j in 0..dim {
                inv_metric[[i, j]] = 1.0 / lambda_sq[i].max(EPS);
            }
        }
        inv_metric
    }
    
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // Poincaré: Γ^k_ij = (2c/(1-c||x||²)) * (δ_ik x_j + δ_jk x_i - δ_ij x_k)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        
        let mut symbols = Vec::new();
        for b in 0..batch_size {
            let coeff = 2.0 * c / (1.0 - c * x_norm_sq[b]).max(EPS);
            let mut gamma = Array2::zeros((dim, dim));
            
            // 대각 근사: i=j=k만 고려
            for i in 0..dim {
                gamma[[i, i]] = coeff * x[[b, i]];
            }
            symbols.push(gamma);
        }
        symbols
    }
    
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::poincare::poincare_distance(x, y, self.curvature)
    }
    
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let lambda_sq = self.conformal_factor(x);
        let dim = x.ncols() as f32;
        lambda_sq.mapv(|l| l.powf(dim))
    }
    
    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

/// 로렌츠 (Hyperboloid) 메트릭
/// Minkowski inner product: ⟨u,v⟩ = u₀v₀ - Σᵢ uᵢvᵢ
pub struct LorentzMetric {
    pub curvature: f32,
}

impl LorentzMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }
}

impl MetricTensor for LorentzMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Minkowski metric: diag(1, -1, -1, ...)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));
        
        for i in 0..batch_size {
            metric[[i, 0]] = 1.0;
            for j in 1..dim {
                metric[[i, j]] = -1.0;
            }
        }
        metric
    }
    
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Minkowski 메트릭은 자기역원
        self.compute_metric(x)
    }
    
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // 민코프스키 공간에서 크리스토펠 기호는 0 (평탄)
        let batch_size = x.nrows();
        let dim = x.ncols();
        vec![Array2::zeros((dim, dim)); batch_size]
    }
    
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::lorentz::lorentz_distance(x, y, self.curvature)
    }
    
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        Array1::from_elem(x.nrows(), -1.0)  // det(η) = -1
    }
    
    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

/// Klein 메트릭 (projective model)
pub struct KleinMetric {
    pub curvature: f32,
}

impl KleinMetric {
    pub fn new(curvature: f32) -> Self {
        Self { curvature }
    }
}

impl MetricTensor for KleinMetric {
    fn compute_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        // Klein: g_ij = (1/(1-c||x||²)) * (δ_ij + c xᵢxⱼ/(1-c||x||²))
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let factor = (1.0 - c * &x_norm_sq).mapv(|v| 1.0 / v.max(EPS));
        
        // 대각 근사: g_ii = factor * (1 + c x_i²/(1-c||x||²))
        let batch_size = x.nrows();
        let dim = x.ncols();
        let mut metric = Array2::zeros((batch_size, dim));
        
        for i in 0..batch_size {
            for j in 0..dim {
                metric[[i, j]] = factor[i] * (1.0 + c * x[[i, j]] * x[[i, j]] * factor[i]);
            }
        }
        metric
    }
    
    fn compute_inverse_metric(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let metric = self.compute_metric(x);
        metric.mapv(|g| 1.0 / g.max(EPS))
    }
    
    fn christoffel_symbols(&self, x: &ArrayView2<f32>) -> Vec<Array2<f32>> {
        // Klein 모델의 크리스토펠 기호 (대각 근사)
        let batch_size = x.nrows();
        let dim = x.ncols();
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        
        let mut symbols = Vec::new();
        for b in 0..batch_size {
            let denom = (1.0 - c * x_norm_sq[b]).max(EPS);
            let coeff = c / denom;
            let mut gamma = Array2::zeros((dim, dim));
            
            for i in 0..dim {
                gamma[[i, i]] = coeff * x[[b, i]];
            }
            symbols.push(gamma);
        }
        symbols
    }
    
    fn distance(&self, x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
        crate::layers::klein::klein_distance(x, y, self.curvature)
    }
    
    fn determinant(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let c = self.curvature;
        let x_norm_sq = crate::ops::norm_sq_batched(x);
        let dim = x.ncols() as f32;
        let factor = 1.0 - c * &x_norm_sq;
        factor.mapv(|f| f.powf(-dim))
    }
    
    fn curvature(&self) -> f32 {
        -self.curvature
    }
}

// 유틸리티 함수
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x  // 수치 안정성
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 메트릭 타입 열거형 (런타임 선택용)
pub enum MetricType {
    Diagonal(DiagonalMetric),
    Poincare(PoincareMetric),
    Lorentz(LorentzMetric),
    Klein(KleinMetric),
}

impl MetricType {
    pub fn as_trait(&self) -> &dyn MetricTensor {
        match self {
            MetricType::Diagonal(m) => m,
            MetricType::Poincare(m) => m,
            MetricType::Lorentz(m) => m,
            MetricType::Klein(m) => m,
        }
    }
    
    pub fn as_trait_mut(&mut self) -> &mut dyn MetricTensor {
        match self {
            MetricType::Diagonal(m) => m,
            MetricType::Poincare(m) => m,
            MetricType::Lorentz(m) => m,
            MetricType::Klein(m) => m,
        }
    }
}

