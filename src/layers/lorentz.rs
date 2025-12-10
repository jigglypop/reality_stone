// 순수 로렌츠 구현 (Poincaré 폴백 없음)
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::ops::{batch::EPS, norm_sq_batched};

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(EPS).sqrt()
}

#[inline]
fn safe_acosh(x: f32) -> f32 {
    (x.max(1.0 + EPS)).acosh()
}

/// 로렌츠 민코프스키 내적 (Minkowski Inner Product)
///
/// <u,v>_L = u0*v0 - u1*v1 - ... - un*vn
pub fn lorentz_inner(u: &ArrayView2<f32>, v: &ArrayView2<f32>) -> Array1<f32> {
    let batch_size = u.nrows();
    let mut result = Array1::zeros(batch_size);

    result
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, inner)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // 민코프스키 내적 계산
            *inner = u_row[0] * v_row[0];
            for j in 1..u_row.len() {
                *inner -= u_row[j] * v_row[j];
            }
        });

    result
}

/// 원점 O = (1/√c, 0, ..., 0) 에서의 지수 맵 (Exponential Map)
///
/// 접벡터 u (R^d)를 쌍곡면(Hyperboloid) 위의 점으로 매핑합니다.
pub fn lorentz_exp0_space(u: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = u.nrows();
    let dim = u.ncols();
    let sqrtc = c.sqrt();
    let u_norm = norm_sq_batched(u).mapv(f32::sqrt);
    let s = u_norm.mapv(|v| sqrtc * v);
    let mut out = Array2::<f32>::zeros((batch, dim + 1));

    // 시간 성분 (Time component)
    {
        let mut tcol = out.slice_mut(s![.., 0..1]);
        let mut idx = 0;
        for mut row in tcol.rows_mut() {
            let sv = s[idx];
            row[[0]] = sv.cosh() / sqrtc;
            idx += 1;
        }
    }
    // 공간 성분 (Space components)
    for i in 0..batch {
        let sv = s[i];
        let scale = if sv.abs() < 1e-6 {
            1.0 / sqrtc
        } else {
            sv.sinh() / (sv * sqrtc)
        };
        for j in 0..dim {
            out[[i, j + 1]] = u[[i, j]] * scale;
        }
    }
    out
}

/// lorentz_exp0_space의 정확한 역전파 (Gradient)
///
/// 입력 u(접벡터)에 대한 그라디언트를 반환합니다.
pub fn lorentz_exp0_space_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    c: f32,
) -> Array2<f32> {
    let batch = u.nrows();
    let d = u.ncols();
    let sqrtc = c.sqrt();
    let mut grad_input = Array2::<f32>::zeros(u.raw_dim());

    for i in 0..batch {
        // r = ||u||, s = sqrt(c) * r
        let mut r_sq = 0.0f32;
        for j in 0..d {
            r_sq += u[[i, j]] * u[[i, j]];
        }
        let r = r_sq.sqrt();
        let s = sqrtc * r;

        // f(s) = sinh(s)/(s*sqrt(c))
        let f = if s.abs() < 1e-6 {
            1.0 / sqrtc
        } else {
            s.sinh() / (s * sqrtc)
        };

        // f'(s) = (cosh(s)*s - sinh(s)) / (s^2 * sqrt(c))
        let fp = if s.abs() < 1e-6 {
            // s가 작을 때의 극한값 처리: g'(s) ~ s/3, f'(s) -> 0
            0.0
        } else {
            (s.cosh() * s - s.sinh()) / (s * s * sqrtc)
        };

        // ds/du_k = sqrt(c) * u_k / r
        let inv_r = if r < 1e-6 { 0.0 } else { 1.0 / r };

        // 출력 그라디언트 수집: 시간 성분 g_t, 공간 성분 g_s
        let g_t = grad_output[[i, 0]];
        let sinh_s = s.sinh();

        // dot(g_s, u)
        let mut g_s_dot_u = 0.0f32;
        for j in 0..d {
            g_s_dot_u += grad_output[[i, j + 1]] * u[[i, j]];
        }

        // 차원별 그라디언트 계산
        for k in 0..d {
            let u_k = u[[i, k]];
            let dsduk = sqrtc * u_k * inv_r;
            let term_from_space = f * grad_output[[i, k + 1]] + (g_s_dot_u * fp * dsduk);
            let term_from_time = g_t * (sinh_s * u_k * inv_r);
            grad_input[[i, k]] = term_from_space + term_from_time;
        }
    }

    grad_input
}

/// 원점에서의 로그 맵 (Logarithmic Map)
///
/// 쌍곡면 위의 점(시간+공간)을 원점의 접공간(R^d)으로 매핑합니다.
pub fn lorentz_log0_space(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = x.nrows();
    let dim = x.ncols() - 1;
    let sqrtc = c.sqrt();
    let mut out = Array2::<f32>::zeros((batch, dim));
    for i in 0..batch {
        let x0 = x[[i, 0]];
        // s = arcosh(√c x0)
        let s = (sqrtc * x0).acosh();
        let denom = s.sinh().max(EPS);
        let scale = if s.abs() < 1e-6 {
            1.0
        } else {
            s / (denom * sqrtc)
        };
        for j in 0..dim {
            out[[i, j]] = x[[i, j + 1]] * scale;
        }
    }
    out
}

/// lorentz_log0_space의 정확한 역전파 (Gradient)
///
/// 입력 x(시간+공간)에 대한 그라디언트를 반환합니다.
pub fn lorentz_log0_space_backward(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
) -> Array2<f32> {
    let batch = x.nrows();
    let dim = x.ncols();
    let space_dim = dim - 1;
    let sqrtc = c.sqrt();
    let mut grad_input = Array2::<f32>::zeros(x.raw_dim());

    for i in 0..batch {
        let x0 = x[[i, 0]];
        // s = acosh( sqrt(c) * x0 )
        let ax0 = (sqrtc * x0).max(1.0 + EPS);
        let s = ax0.acosh();
        let sinh_s = s.sinh().max(EPS);
        let cosh_s = s.cosh();

        // scale = s / (sinh(s) * sqrt(c))
        let scale = s / (sinh_s * sqrtc);

        // d scale / d x0 = (sinh(s) - s cosh(s)) / sinh(s)^3
        let dscale_dx0 = (sinh_s - s * cosh_s) / (sinh_s * sinh_s * sinh_s);

        // g_xspace = grad_output_space * scale
        for j in 0..space_dim {
            grad_input[[i, j + 1]] = grad_output[[i, j]] * scale;
        }

        // 시간 성분: scale을 통해서만 영향 받음
        let mut dot_gspace_xspace = 0.0f32;
        for j in 0..space_dim {
            dot_gspace_xspace += grad_output[[i, j]] * x[[i, j + 1]];
        }
        grad_input[[i, 0]] = dot_gspace_xspace * dscale_dx0;
    }

    grad_input
}

/// 로렌츠 거리 (Lorentz Distance)
///
/// cosh(√c d) = c <u,v>_L
pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let inner = lorentz_inner(u, v);
    let sqrtc = c.sqrt();
    inner.mapv(|x| safe_acosh((c * x).max(1.0 + EPS)) / sqrtc)
}

/// 로렌츠 덧셈 (Lorentz Addition)
///
/// 자이로벡터 공간(Gyrovector space)에서의 덧셈 연산을 수행합니다.
pub fn lorentz_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // 내적 계산 (Inner products)
            let mut uu = u_row[0] * u_row[0];
            let mut vv = v_row[0] * v_row[0];
            let mut uv = u_row[0] * v_row[0];

            for j in 1..dim {
                uu -= u_row[j] * u_row[j];
                vv -= v_row[j] * v_row[j];
                uv -= u_row[j] * v_row[j];
            }

            let beta_u = (-uu / c).max(EPS);
            let beta_v = (-vv / c).max(EPS);
            let gamma_u = 1.0 / safe_sqrt(beta_u);
            let gamma_v = 1.0 / safe_sqrt(beta_v);
            let gamma_uv = -uv / (c * (beta_u * beta_v).sqrt());

            for j in 0..dim {
                let denom_u = (1.0 + gamma_u).max(EPS);
                let denom_v = (1.0 + gamma_v).max(EPS);
                row[j] = gamma_uv * (gamma_u * u_row[j] / denom_u + gamma_v * v_row[j] / denom_v)
                    + u_row[j]
                    + v_row[j];
            }
        });

    result
}

/// 로렌츠 스칼라 곱 (Lorentz Scalar Multiplication)
pub fn lorentz_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let time_comp = u_row[0];

            let mut space_norm_sq = 0.0;
            for j in 1..dim {
                space_norm_sq += u_row[j] * u_row[j];
            }

            // 쌍곡면 제약조건: time^2 - ||x||^2 = 1/c
            let denom = (time_comp * time_comp - 1.0 / c).max(EPS);
            let norm = (space_norm_sq / denom).sqrt();
            let theta = norm.min(1.0 - EPS).atanh() * r;
            let scale = theta.tanh() / norm.max(EPS);

            // 공간 성분 설정
            let mut scaled_space_norm_sq = 0.0;
            for j in 1..dim {
                row[j] = u_row[j] * scale;
                scaled_space_norm_sq += row[j] * row[j];
            }
            // 시간 성분 재계산: x0 = sqrt(1/c + ||x||^2)
            row[0] = (1.0 / c + scaled_space_norm_sq).sqrt();
        });

    result
}

/// lorentz_scalar의 정확한 역전파 (Gradient)
///
/// 입력 u(time + space)에 대한 그라디언트를 반환합니다.
pub fn lorentz_scalar_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let space_dim = dim - 1;
    let mut grad_input = Array2::<f32>::zeros(u.raw_dim());

    for i in 0..batch_size {
        let t = u[[i, 0]];
        // 공간 벡터 s
        let mut space_norm_sq = 0.0f32;
        for j in 0..space_dim {
            let v = u[[i, j + 1]];
            space_norm_sq += v * v;
        }

        // 순전파 계산 재현
        let denom = (t * t - 1.0 / c).max(EPS);
        let ns = space_norm_sq.sqrt();
        let ns_safe = ns.max(EPS);
        let sqrt_denom = denom.sqrt();
        let norm = (ns / sqrt_denom).max(0.0);

        let norm_clamp_top = 1.0 - EPS;
        let scn = norm.min(norm_clamp_top);
        let alpha = scn.atanh();
        let theta = r * alpha;
        let beta = theta.tanh();
        let scale = if norm < EPS { r } else { beta / norm };

        // s' 와 t'
        let mut s_prime_sq = 0.0f32;
        for j in 0..space_dim {
            let sp = u[[i, j + 1]] * scale;
            s_prime_sq += sp * sp;
        }
        let t_prime = (1.0 / c + s_prime_sq).sqrt();

        // s' 에 대한 유효 그라디언트 (시간 성분 경로 포함)
        let g_tprime = grad_output[[i, 0]];
        // d t' / d s'_j = s'_j / t' 미리 계산
        // dot(g_sprime_eff, s) 누적
        let mut dot_gs_eff_s = 0.0f32;

        let mut g_sprime_eff: Vec<f32> = vec![0.0; space_dim];
        for j in 0..space_dim {
            let s_j = u[[i, j + 1]];
            let s_prime_j = s_j * scale;
            let g_sprime_j = grad_output[[i, j + 1]];
            let g_eff = g_sprime_j + g_tprime * (s_prime_j / t_prime.max(EPS));
            g_sprime_eff[j] = g_eff;
            dot_gs_eff_s += g_eff * s_j;
        }

        // norm의 미분
        // d norm / d s_k = s_k / (ns * sqrt(denom))
        // d norm / d t = - (t / denom) * norm
        let inv_sqrt_denom = 1.0 / sqrt_denom;
        let dnorm_dt = -(t / denom) * norm;

        // d beta / d norm = r * (1 - beta^2) / (1 - scn^2) * dscn/dnorm
        let dscn_dnorm = if norm <= norm_clamp_top { 1.0 } else { 0.0 };
        let one_minus_beta_sq = 1.0 - beta * beta;
        let one_minus_scn_sq = (1.0 - scn * scn).max(EPS);
        let dbeta_dnorm = r * one_minus_beta_sq / one_minus_scn_sq * dscn_dnorm;

        // d scale / d norm = (norm * dbeta_dnorm - beta) / norm^2
        let dscale_dnorm = if norm < 1e-6 {
            0.0
        } else {
            (norm * dbeta_dnorm - beta) / (norm * norm)
        };

        // d scale / d s_k 및 d scale / d t
        let mut dscale_ds: Vec<f32> = vec![0.0; space_dim];
        for k in 0..space_dim {
            let dnorm_dsk = if ns_safe < 1e-6 {
                0.0
            } else {
                u[[i, k + 1]] * (1.0 / (ns_safe)) * inv_sqrt_denom
            };
            dscale_ds[k] = dscale_dnorm * dnorm_dsk;
        }
        let dscale_dt = dscale_dnorm * dnorm_dt;

        // 최종 그라디언트 누적
        // 공간 성분에 대한 그라디언트
        for k in 0..space_dim {
            let gk = g_sprime_eff[k] * scale + dot_gs_eff_s * dscale_ds[k];
            grad_input[[i, k + 1]] = gk;
        }

        // 시간 성분에 대한 그라디언트
        let mut g_time = 0.0f32;
        g_time += dot_gs_eff_s * dscale_dt;
        grad_input[[i, 0]] = g_time;
    }

    grad_input
}

/// 로렌츠 -> 클라인 변환
pub fn lorentz_to_klein(x: &ArrayView2<f32>, _: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols() - 1;
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x0 = x_row[0].max(EPS);

            for j in 0..dim {
                row[j] = x_row[j + 1] / x0;
            }
        });

    result
}

/// 로렌츠 -> 푸앵카레 변환
pub fn lorentz_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let klein = lorentz_to_klein(x, c);
    crate::layers::klein::klein_to_poincare(&klein.view(), c)
}

/// 로렌츠 모델의 순전파 레이어 계산
///
/// 파라미터 t에 따라 u와 v 사이를 지오데식(Geodesic)으로 보간합니다.
pub fn lorentz_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::<f32>::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let p = u.row(i);
            let q = v.row(i);
            // 민코프스키 내적
            let mut inner = p[0] * q[0];
            for j in 1..dim {
                inner -= p[j] * q[j];
            }
            let theta = safe_acosh((c * inner).max(1.0 + EPS));
            let sinh_theta = theta.sinh().max(EPS);

            // 가중치 w1, w2 계산
            let w1 = if theta.abs() < 1e-6 {
                1.0 - t
            } else {
                ((1.0 - t) * theta).sinh() / sinh_theta
            };
            let w2 = if theta.abs() < 1e-6 {
                t
            } else {
                (t * theta).sinh() / sinh_theta
            };

            // 선형 결합 (시간 성분 포함)
            for j in 0..dim {
                row[j] = w1 * p[j] + w2 * q[j];
            }
        });

    result
}

/// 로렌츠 모델의 역전파 레이어 계산
pub fn lorentz_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut gu = Array2::<f32>::zeros(u.raw_dim());
    let mut gv = Array2::<f32>::zeros(v.raw_dim());

    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        let g = grad_output.row(i);

        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }

        let alpha_arg = (c * inner).max(1.0 + EPS);
        let alpha = alpha_arg.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        // 가중치 w1, w2
        let w1 = if alpha.abs() < 1e-6 {
            1.0 - t
        } else {
            ((1.0 - t) * alpha).sinh() / sinh_alpha
        };
        let w2 = if alpha.abs() < 1e-6 {
            t
        } else {
            (t * alpha).sinh() / sinh_alpha
        };

        // dw/dalpha
        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        // d alpha / d p = (c / sinh(alpha)) * G q  (G는 민코프스키 메트릭)
        let scale = c / sinh_alpha;
        let mut dalpha_dp = vec![0.0f32; dim];
        let mut dalpha_dq = vec![0.0f32; dim];
        dalpha_dp[0] = scale * q[0];
        dalpha_dq[0] = scale * p[0];
        for j in 1..dim {
            dalpha_dp[j] = scale * (-q[j]);
            dalpha_dq[j] = scale * (-p[j]);
        }

        // g dot p, g dot q (유클리드)
        let mut g_dot_p = 0.0f32;
        let mut g_dot_q = 0.0f32;
        for j in 0..dim {
            g_dot_p += g[j] * p[j];
            g_dot_q += g[j] * q[j];
        }

        for j in 0..dim {
            gu[[i, j]] = w1 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dp[j];
            gv[[i, j]] = w2 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dq[j];
        }
    }

    (gu, gv)
}

fn acosh_derivative(z: f32) -> f32 {
    // d/dz acosh(z) = 1 / (sqrt(z-1) * sqrt(z+1))
    let zp = (z + 1.0).max(1.0 + EPS);
    let zm = (z - 1.0).max(EPS);
    1.0 / (zp.sqrt() * zm.sqrt())
}

/// 동적 곡률을 사용한 로렌츠 레이어 순전파
pub fn lorentz_layer_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let y = lorentz_layer_forward(u, v, c, t);
    (y, c)
}

/// 동적 곡률을 사용한 로렌츠 레이어 역전파 (정석 미분)
pub fn lorentz_layer_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &crate::ops::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let (grad_u, grad_v) = lorentz_layer_backward(grad_output, u, v, c, t);

    // Chain Rule을 통한 grad_c 계산
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut grad_c = 0.0f32;
    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        // 민코프스키 내적
        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }
        let z = (c * inner).max(1.0 + EPS);
        let alpha = z.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        // dw/dalpha
        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        // dalpha/dc = (d acosh(z)/dz) * dz/dc, where z = c * inner
        let dalpha_dz = acosh_derivative(z);
        let dz_dc = inner;
        let dalpha_dc = dalpha_dz * dz_dc;

        let dw1_dc = dw1_dalpha * dalpha_dc;
        let dw2_dc = dw2_dalpha * dalpha_dc;

        // dy/dc = dw1_dc * p + dw2_dc * q; grad_c 누적
        for j in 0..dim {
            let d_yj_dc = dw1_dc * p[j] + dw2_dc * q[j];
            grad_c += grad_output[[i, j]] * d_yj_dc;
        }
    }

    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;
    (grad_u, grad_v, grad_kappa)
}

/// 레이어별 곡률을 사용한 로렌츠 레이어 순전파
pub fn lorentz_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let y = lorentz_layer_forward(u, v, c, t);
    (y, c)
}

/// 레이어별 곡률을 사용한 로렌츠 레이어 역전파
pub fn lorentz_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &crate::ops::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let (grad_u, grad_v) = lorentz_layer_backward(grad_output, u, v, c, t);

    // grad_c 누적 (dynamic 버전과 동일한 로직)
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut grad_c = 0.0f32;
    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }
        let z = (c * inner).max(1.0 + EPS);
        let alpha = z.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        let dalpha_dz = acosh_derivative(z);
        let dz_dc = inner;
        let dalpha_dc = dalpha_dz * dz_dc;
        let dw1_dc = dw1_dalpha * dalpha_dc;
        let dw2_dc = dw2_dalpha * dalpha_dc;

        for j in 0..dim {
            let d_yj_dc = dw1_dc * p[j] + dw2_dc * q[j];
            grad_c += grad_output[[i, j]] * d_yj_dc;
        }
    }

    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c * dc_dkappa;
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn lorentz_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_forward_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_backward_cuda(
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

    pub fn lorentz_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn lorentz_layer_forward_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_layer_forward_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn lorentz_layer_backward_cuda(
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
            ffi::lorentz_layer_backward_cuda(
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

/// 푸앵카레 -> 로렌츠 변환
pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let factor = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));

    result
        .slice_mut(s![.., 0..1])
        .assign(&(&factor * (1.0 + c * &x_norm_sq) / c.sqrt()));
    result
        .slice_mut(s![.., 1..])
        .assign(&(&factor * 2.0 * x / c.sqrt()));
    result
}

/// 푸앵카레 -> 로렌츠 변환의 곡률 그라디언트
pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut grad_result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));
    let sqrt_c = c.sqrt();

    // 시간 성분 그라디언트
    let d_time_den_dc = -&x_norm_sq;
    let d_time_num_dc = &x_norm_sq;
    let time_num = 1.0 + c * &x_norm_sq;
    let d_time_dc = (d_time_num_dc * &den - &time_num * d_time_den_dc) / (&den * &den);
    grad_result
        .slice_mut(s![.., 0..1])
        .assign(&(&d_time_dc / sqrt_c - &time_num / (2.0 * c * sqrt_c * &den)));

    // 공간 성분 그라디언트
    let d_factor_dc = &x_norm_sq / (&den * &den);
    grad_result
        .slice_mut(s![.., 1..])
        .assign(&(x * (&d_factor_dc / sqrt_c - 1.0 / (c * sqrt_c * &den))));

    grad_result
}
