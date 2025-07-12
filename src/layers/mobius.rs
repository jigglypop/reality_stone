use crate::layers::utils::{dot_batched, norm_sq_batched, EPS};
use ndarray::{Array2, ArrayView2, Axis};

const BOUNDARY_EPS: f32 = 1e-5;
const MIN_DENOMINATOR: f32 = 1e-6;

pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;

    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;

    coeff_u * u + coeff_v * v
}

pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    
    if c.abs() < EPS {
        // c = 0: 유클리드 경우
        return Array2::from_elem(u.dim(), r) * u;
    }
    
    // 음수 곡률도 처리 가능하도록 수정
    // sqrt(c) * norm이 1보다 작아야 atanh가 정의됨
    let sqrt_c_norm = if c > 0.0 {
        (c.sqrt() * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS))
    } else {
        // c < 0일 때는 sqrt(|c|) * norm을 사용하고, atanh 대신 atan 사용
        ((-c).sqrt() * &norm_clamped)
    };
    
    let scale = if c > 0.0 {
        // 양수 곡률: 원래 공식
        let alpha = sqrt_c_norm.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        beta / (c.sqrt() * &norm_clamped)
    } else {
        // 음수 곡률: atanh(i*x) = i*atan(x), tanh(i*x) = i*tan(x)
        // 결과적으로 허수 i가 약분되어 실수 결과를 얻음
        let alpha = sqrt_c_norm.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        beta / ((-c).sqrt() * &norm_clamped)
    };
    
    scale * u
}

pub fn mobius_scalar_grad_c(
    u: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    
    if c.abs() < EPS {
        // c = 0: gradient is 0
        return Array2::zeros(u.dim());
    }
    
    if c > 0.0 {
        // 양수 곡률
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        
        // d(sqrt(c))/dc = 0.5/sqrt(c)
        let d_sqrt_c_dc = 0.5 / sqrt_c;
        
        // d(alpha)/d(scn) = 1/(1 - scn^2)
        let d_alpha_dscn = 1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS));
        
        // d(beta)/d(alpha) = r * (1 - tanh^2(r*alpha))
        let tanh_r_alpha = (r * &alpha).mapv(|v| v.tanh());
        let d_beta_dalpha = r * (1.0 - &tanh_r_alpha * &tanh_r_alpha);
        // Chain rule
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_c_dc;
        let d_scale_dc = (&d_beta_dc * sqrt_c - &beta * d_sqrt_c_dc) / (c * &norm_clamped);
        &d_scale_dc * u
    } else {
        // 음수 곡률
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &norm_clamped;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        // d(sqrt(|c|))/dc = -0.5/sqrt(|c|) (c가 음수이므로)
        let d_sqrt_abs_c_dc = -0.5 / sqrt_abs_c;
        // d(alpha)/d(scn) = 1/(1 + scn^2)
        let d_alpha_dscn = 1.0 / (1.0 + &scn * &scn);
        // d(beta)/d(alpha) = r * (1 + tan^2(r*alpha))
        let tan_r_alpha = (r * &alpha).mapv(|v| v.tan());
        let d_beta_dalpha = r * (1.0 + &tan_r_alpha * &tan_r_alpha);
        // Chain rule
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_abs_c_dc;
        // d(scale)/dc
        let d_scale_dc = (&d_beta_dc * sqrt_abs_c - &beta * d_sqrt_abs_c_dc) / ((-c) * &norm_clamped);
        &d_scale_dc * u
    }
}

// 동적 곡률 구조체
#[derive(Debug, Clone)]
pub struct DynamicCurvature {
    pub kappa: f32,
    pub c_min: f32,
    pub c_max: f32,
}

impl DynamicCurvature {
    pub fn new(kappa: f32, c_min: f32, c_max: f32) -> Self {
        Self { kappa, c_min, c_max }
    }
    
    pub fn compute_c(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }
    
    pub fn compute_dc_dkappa(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}

#[derive(Debug, Clone)]
pub struct LayerWiseDynamicCurvature {
    pub kappas: Vec<f32>,
    pub c_min: f32,
    pub c_max: f32,
}

impl LayerWiseDynamicCurvature {
    pub fn new(num_layers: usize, c_min: f32, c_max: f32) -> Self {
        Self { 
            kappas: vec![0.0; num_layers],
            c_min, 
            c_max 
        }
    }
    
    pub fn from_kappas(kappas: Vec<f32>, c_min: f32, c_max: f32) -> Self {
        Self { kappas, c_min, c_max }
    }
    
    pub fn compute_c(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }
    
    pub fn compute_dc_dkappa(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}


pub fn mobius_add_grad_c(
    u: &ArrayView2<f32>, 
    v: &ArrayView2<f32>, 
    c: f32
) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    
    let num = (1.0 + 2.0 * c * &uv + c * &v2) * u + (1.0 - c * &u2) * v;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    
    let dnum_dc = (2.0 * &uv + &v2) * u - &u2 * v;
    
    let dden_dc = 2.0 * &uv + 2.0 * c * &u2 * &v2;
    
    let result = (dnum_dc * &den - &num * &dden_dc) / (&den * &den);
    
    result
}

// 동적 곡률을 사용한 Mobius 덧셈
pub fn mobius_add_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let result = mobius_add(u, v, c);
    (result, c)
}

// 동적 곡률 Mobius 덧셈의 backward pass
pub fn mobius_add_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;
    
    use crate::layers::poincare::mobius_add_vjp;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    
    (grad_u, grad_v, grad_kappa)
}

pub fn mobius_add_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let result = mobius_add(u, v, c);
    (result, c)
}

pub fn mobius_add_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c * dc_dkappa;
    
    use crate::layers::poincare::mobius_add_vjp;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        #[link(name = "kernel_mobius", kind="static")]
        extern "C" {
            pub fn mobius_add_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn mobius_scalar_cuda(
                out: *mut f32,
                u: *const f32,
                c: f32,
                r: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn mobius_add_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_add_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn mobius_scalar_cuda(
        out: *mut f32,
        u: *const f32,
        c: f32,
        r: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_scalar_cuda(out, u, c, r, batch_size, dim);
        }
    }
} 