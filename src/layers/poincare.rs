use crate::ops::{batch::EPS, dot_batched, mobius, norm_sq_batched};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

// 매직 넘버 사용을 피하기 위한 공통 상수
const BOUNDARY_EPS: f32 = 1e-5;

/// 푸앵카레 공 레이어의 역전파 (Backward Pass)
///
/// 입력:
/// - grad_output: 출력에 대한 그라디언트
/// - u, v: 순전파 시 입력 벡터들
/// - c: 곡률
/// - t: 보간 파라미터 (0.0 ~ 1.0)
///
/// 출력:
/// - (grad_u, grad_v): 입력 u, v에 대한 그라디언트
pub fn poincare_ball_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    // 뫼비우스 덧셈의 VJP
    let (grad_u_prime, grad_v_prime) =
        mobius::mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);

    // 스칼라 곱의 VJP
    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);

    (grad_u, grad_v)
}

/// 푸앵카레 거리 (Poincaré Distance) 계산
///
/// d(u, v) = (2/√c) * atanh(√(c * ||u-v||² / ((1-c||u||²)(1-c||v||²))))
pub fn poincare_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, boundary_eps: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let norm_sq_diff = (&u2 + &v2 - 2.0 * &uv).mapv_into(|val| val.max(0.0));
    let den = (1.0 - c * &u2) * (1.0 - c * &v2);
    
    // 경계 근처에서의 수치적 안정성을 위해 분모 제한
    let den_clamped = den.mapv_into(|val| val.max(boundary_eps));

    let frac = norm_sq_diff / den_clamped;
    // 수치 안정성을 고려한 변형 공식 사용:
    // arg = √(c * frac / (1 + c * frac))
    // d = (2/sqrt(c)) * atanh(sqrt(delta / (2 + delta))) 와 유사
    frac.mapv_into(|val| {
        let cf = c * val;
        let arg = (cf / (1.0 + cf)).sqrt().min(1.0 - boundary_eps);
        (2.0 / sqrtc) * arg.atanh()
    })
}

/// 푸앵카레 공 -> 로렌츠(Lorentz) 모델 변환
///
/// x_L = ( (1+c|x|^2)/(1-c|x|^2) * 1/√c,  2x / (1-c|x|^2) * 1/√c )
pub fn poincare_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv_into(|v| v.max(EPS));
    let sqrtc = c.sqrt();

    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let time_component = (1.0 + c * &x_norm_sq) / (&den * sqrtc);
    let space_components = (2.0 * x) / (&den * sqrtc);

    result.slice_mut(s![.., 0..1]).assign(&time_component);
    result.slice_mut(s![.., 1..]).assign(&space_components);
    result
}

/// 푸앵카레 공 -> 클라인(Klein) 모델 변환
///
/// x_K = 2x / (1 + c|x|^2)
pub fn poincare_to_klein(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));
    (2.0 * x) / &den_clamped
}

/// 푸앵카레 공의 지수 맵 (Exponential Map)
///
/// 점 x에서의 접벡터 v를 푸앵카레 공 위의 점으로 매핑합니다.
/// Exp_x(v) = x ⊕_c (tanh( (λ_x * √c * |v|)/2 ) * v / (√c * |v|))
pub fn poincare_exp_at(x: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, _: f32) -> Array2<f32> {
    // 컨포멀 팩터 λ_x = 2 / (1 - c ||x||^2)
    let x2 = norm_sq_batched(x).insert_axis(Axis(1));
    let one_minus_cx2 = (1.0 - c * &x2).mapv(|z| z.max(EPS));
    let lambda_x = 2.0 / &one_minus_cx2;

    // |v| 계산 및 안전한 스케일링
    let vnorm = norm_sq_batched(v).mapv(f32::sqrt).insert_axis(Axis(1));
    let vnorm_safe = vnorm.mapv(|z| z.max(EPS));

    if c.abs() < EPS {
        // 유클리드 극한: Exp ≈ x + v
        return x + v;
    }

    let sqrtc = c.sqrt();
    // u = tanh( (λ_x * √c * |v|)/2 ) / (√c * |v|) * v
    let arg = (&lambda_x * sqrtc * &vnorm_safe) * 0.5;
    let coeff = arg.mapv(|a| a.tanh()) / (sqrtc * &vnorm_safe);
    let u = &coeff * v;
    
    // 뫼비우스 덧셈으로 이동
    mobius::mobius_add(x, &u.view(), c)
}

/// 푸앵카레 공의 로그 맵 (Logarithmic Map)
///
/// 두 점 x, y에 대해 x에서 y로 향하는 접벡터를 계산합니다.
/// Log_x(y) = (2 / (√c λ_x)) * atanh( √c |(-x) ⊕_c y| ) * ((-x) ⊕_c y) / |(-x) ⊕_c y|
pub fn poincare_log_at(x: &ArrayView2<f32>, y: &ArrayView2<f32>, c: f32, boundary_eps: f32) -> Array2<f32> {
    // 컨포멀 팩터 λ_x
    let x2 = norm_sq_batched(x).insert_axis(Axis(1));
    let one_minus_cx2 = (1.0 - c * &x2).mapv(|z| z.max(EPS));
    let lambda_x = 2.0 / &one_minus_cx2;

    if c.abs() < EPS {
        // 유클리드 극한: Log ≈ y - x
        return y - x;
    }

    // z = (-x) ⊕_c y (x를 원점으로 이동시켰을 때의 y의 위치)
    let neg_x = -x;
    let z = mobius::mobius_add(&neg_x.view(), y, c);
    let znorm = norm_sq_batched(&z.view())
        .mapv(f32::sqrt)
        .insert_axis(Axis(1));
    
    // 수치적 안정성을 위해 norm 클리핑
    let znorm_clip = znorm.mapv(|r| r.min(1.0 - boundary_eps).max(EPS));

    let sqrtc = c.sqrt();
    let atanh_term = (&znorm_clip * sqrtc).mapv(|u| u.atanh());
    let scale = (2.0 / (sqrtc * &lambda_x)) * &atanh_term / &znorm_clip;
    &scale * &z
}

/// 푸앵카레 공 레이어 (순전파)
///
/// 두 입력 u, v를 뫼비우스 덧셈으로 결합합니다.
/// 파라미터 t에 의해 가중치가 조절됩니다: ((1-t) ⊗ u) ⊕ (t ⊗ v)
pub fn poincare_ball_layer(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}

/// 동적 곡률을 사용하는 푸앵카레 레이어
pub fn poincare_ball_layer_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_dynamic(&u_prime.view(), &v_prime.view(), dynamic_c);
    (result, c)
}

/// 동적 곡률 푸앵카레 레이어의 역전파
pub fn poincare_ball_layer_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    let (grad_u_prime, grad_v_prime, grad_kappa) = mobius::mobius_add_dynamic_backward(
        grad_output,
        &u_prime.view(),
        &v_prime.view(),
        dynamic_c,
    );

    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);

    (grad_u, grad_v, grad_kappa)
}

/// 레이어별 동적 곡률을 사용하는 푸앵카레 레이어
pub fn poincare_ball_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_layerwise(
        &u_prime.view(),
        &v_prime.view(),
        layer_curvatures,
        layer_idx,
    );
    (result, c)
}

/// 레이어별 동적 곡률 푸앵카레 레이어의 역전파
pub fn poincare_ball_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    
    let (grad_u_prime, grad_v_prime) =
        mobius::mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);
        
    let grad_u = mobius::mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius::mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);
    
    // 곡률 c에 대한 그라디언트 계산 (Chain Rule)
    let grad_c_from_add_tensor = mobius::mobius_add_grad_c(&u_prime.view(), &v_prime.view(), c);
    let grad_c_add = (grad_output * &grad_c_from_add_tensor).sum();
    
    let grad_c_from_scalar_u_tensor = mobius::mobius_scalar_grad_c(u, c, 1.0 - t);
    let grad_c_scalar_u = (&grad_u_prime * &grad_c_from_scalar_u_tensor).sum();
    
    let grad_c_from_scalar_v_tensor = mobius::mobius_scalar_grad_c(v, c, t);
    let grad_c_scalar_v = (&grad_v_prime * &grad_c_from_scalar_v_tensor).sum();
    
    let grad_c_total = grad_c_add + grad_c_scalar_u + grad_c_scalar_v;
    
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c_total * dc_dkappa;
    
    (grad_u, grad_v, grad_kappa)
}

/// 리만 아담 (Riemannian Adam) 옵티마이저 스텝
///
/// 푸앵카레 공 위에서의 파라미터 업데이트를 수행합니다.
/// 1. 그라디언트를 리만 그라디언트로 변환 (스케일링)
/// 2. 모멘텀 업데이트
/// 3. 접공간에서의 업데이트 방향 계산
/// 4. 지수 맵(Exponential Map)을 통한 파라미터 갱신
pub fn poincare_riemannian_adam_step(
    x: &ArrayView2<f32>,
    grad: &ArrayView2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    step: u64,
    c: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    max_norm_eps: f32,
) -> Array2<f32> {
    let mut g_r: Array2<f32>;
    
    // 리만 그라디언트 변환: grad_R = grad_E / (lambda_x)^2
    if c.abs() < EPS {
        g_r = grad.to_owned();
    } else {
        let norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
        let one_minus_cx2 = (1.0 - c * &norm_sq).mapv(|z| z.max(EPS));
        let lambda = 2.0 / &one_minus_cx2;
        let inv_lambda_sq = 1.0 / (&lambda * &lambda);
        
        g_r = grad.to_owned();
        for (mut row, factor) in g_r
            .axis_iter_mut(Axis(0))
            .zip(inv_lambda_sq.axis_iter(Axis(0)))
        {
            let f = factor[0];
            for val in row.iter_mut() {
                *val *= f;
            }
        }
    }
    
    let one_minus_b1 = 1.0 - beta1;
    let one_minus_b2 = 1.0 - beta2;

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_r
    ndarray::Zip::from(&mut *m)
        .and(&g_r)
        .for_each(|m_elt, g_elt| {
            *m_elt = beta1 * *m_elt + one_minus_b1 * *g_elt;
        });

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_r^2
    ndarray::Zip::from(&mut *v)
        .and(&g_r)
        .for_each(|v_elt, g_elt| {
            *v_elt = beta2 * *v_elt + one_minus_b2 * (*g_elt * *g_elt);
        });

    // Bias correction
    let t = step as f32;
    let bias_c1 = 1.0 - beta1.powf(t);
    let bias_c2 = 1.0 - beta2.powf(t);
    let m_hat = m.mapv(|val| val / bias_c1);
    let v_hat = v.mapv(|val| val / bias_c2);

    // 업데이트 벡터 u (접공간)
    let mut u = m_hat.clone();
    ndarray::Zip::from(&mut u)
        .and(&v_hat)
        .for_each(|u_elt, v_elt| {
            *u_elt = -*u_elt * lr / (v_elt.sqrt() + eps);
        });

    if c.abs() < EPS {
        &x.to_owned() + &u
    } else {
        // 사용자 정의 max_norm_eps 또는 기본 안전값 사용
        let safe_eps = if max_norm_eps > 0.0 { max_norm_eps } else { BOUNDARY_EPS };
        // Exponential Map으로 업데이트 적용
        let x_new = poincare_exp_at(x, &u.view(), c, safe_eps);
        
        // 투영(Project) 로직 인라인 구현 (공 밖으로 나가지 않도록)
        let mut out = x_new.to_owned();
        let mut norms = norm_sq_batched(&out.view())
            .mapv(f32::sqrt)
            .insert_axis(Axis(1));
        let radius = if c > 0.0 { 1.0 / c.sqrt() } else { 1.0 };
        let max_norm = radius - safe_eps;
        
        for (mut row, mut norm) in out.axis_iter_mut(Axis(0)).zip(norms.axis_iter_mut(Axis(0))) {
            let n = norm[0].max(EPS);
            if n > max_norm {
                let scale = max_norm / n;
                row *= scale;
                norm[0] = max_norm;
            }
        }
        out
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn poincare_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                boundary_eps: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_backward_cuda(
                grad_output: *const f32,
                u: *const f32,
                v: *const f32,
                grad_u: *mut f32,
                grad_v: *mut f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn poincare_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        boundary_eps: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_distance_cuda(out, u, v, c, boundary_eps, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_ball_layer_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_backward_cuda(
        grad_output: *const f32,
        u: *const f32,
        v: *const f32,
        grad_u: *mut f32,
        grad_v: *mut f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_ball_layer_backward_cuda(
                grad_output,
                u,
                v,
                grad_u,
                grad_v,
                c,
                t,
                batch_size,
                dim,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::lorentz;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_mobius_add_identity() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2]]);
        let z = arr2(&[[0.0, 0.0]]);
        let result = mobius::mobius_add(&x.view(), &z.view(), c);
        assert_relative_eq!(result, x, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_to_lorentz_and_back() {
        let c = 1.0;
        let x_poincare = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let x_lorentz = poincare_to_lorentz(&x_poincare.view(), c);
        let x_poincare_restored = lorentz::lorentz_to_poincare(&x_lorentz.view(), c);

        assert_relative_eq!(x_poincare, x_poincare_restored, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_ball_layer_interpolation() {
        let c = 1.0;
        let u = arr2(&[[0.5, 0.5]]);
        let v = arr2(&[[-0.5, -0.5]]);

        // t=0 이면 u와 같아야 함
        let result_t0 = poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
        assert_relative_eq!(result_t0, u, epsilon = EPSILON);

        // t=1 이면 v와 같아야 함
        let result_t1 = poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
        assert_relative_eq!(result_t1, v, epsilon = EPSILON);

        // t=0.5 대칭성
        let result_t05 = poincare_ball_layer(&u.view(), &v.view(), c, 0.5);
        let result_t05_sym = poincare_ball_layer(&v.view(), &u.view(), c, 0.5);
        assert_relative_eq!(result_t05, result_t05_sym, epsilon = 1e-5);
    }

    #[test]
    fn test_distance_is_zero_for_same_point() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let dist = poincare_distance(&x.view(), &x.view(), c, 1e-5);

        for val in dist.iter() {
            // 수치적 클램프로 인해 0이 아닌 매우 작은 값이 나올 수 있음
            assert!((*val).abs() < 1e-3);
        }
    }

    #[test]
    fn test_poincare_to_klein_then_back_shape_and_finiteness() {
        let c = 0.7_f32;
        let x = arr2(&[[0.1, -0.2], [0.3, 0.1]]);
        let k = poincare_to_klein(&x.view(), c);
        assert_eq!(k.ncols(), x.ncols());
        assert_eq!(k.nrows(), x.nrows());
        assert!(k.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_riemannian_adam_matches_euclidean_when_c_zero() {
        let x = arr2(&[[0.5f32, -0.3f32]]);
        let grad = arr2(&[[0.5f32, -0.3f32]]);
        let mut m = Array2::<f32>::zeros((1, 2));
        let mut v = Array2::<f32>::zeros((1, 2));
        let step = 1;
        let c = 0.0;
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let x_view = x.view();
        let grad_view = grad.view();
        let x_new = poincare_riemannian_adam_step(
            &x_view,
            &grad_view,
            &mut m,
            &mut v,
            step,
            c,
            lr,
            beta1,
            beta2,
            eps,
            1e-5,
        );
        let mut m_e = Array2::<f32>::zeros((1, 2));
        let mut v_e = Array2::<f32>::zeros((1, 2));
        let g = grad.clone();
        m_e = m_e * beta1 + &g * (1.0 - beta1);
        v_e = v_e * beta2 + &g.mapv(|x| x * x) * (1.0 - beta2);
        let m_hat = &m_e / (1.0 - beta1);
        let v_hat = &v_e / (1.0 - beta2);
        let mut u = m_hat.clone();
        ndarray::Zip::from(&mut u)
            .and(&v_hat)
            .for_each(|u_elt, v_elt| {
                *u_elt = -*u_elt * lr / (v_elt.sqrt() + eps);
            });
        let x_expected = &x + &u;
        assert_relative_eq!(x_new, x_expected, epsilon = 1e-6);
    }

    #[test]
    fn test_riemannian_adam_poincare_stays_inside_ball() {
        let x = arr2(&[[0.5f32, 0.4f32]]);
        let grad = arr2(&[[0.5f32, 0.4f32]]);
        let mut m = Array2::<f32>::zeros((1, 2));
        let mut v = Array2::<f32>::zeros((1, 2));
        let step = 1;
        let c = 1.0;
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let x_view = x.view();
        let grad_view = grad.view();
        let x_new = poincare_riemannian_adam_step(
            &x_view,
            &grad_view,
            &mut m,
            &mut v,
            step,
            c,
            lr,
            beta1,
            beta2,
            eps,
            1e-5,
        );
        let norms = norm_sq_batched(&x_new.view());
        let n = norms[0].sqrt();
        assert!(n < 1.0 - 1e-3);
    }
}
