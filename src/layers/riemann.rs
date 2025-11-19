use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::layers::poincare::{poincare_exp_at, poincare_log_at};
use crate::ops::project_to_ball;

fn zeros_like(x: &ArrayView2<f32>) -> Array2<f32> {
    Array2::<f32>::zeros((x.nrows(), x.ncols()))
}

/// Riemann low-rank forward on Poincaré ball (tangent at origin):
/// y = Exp_0( ((Log_0(Proj(x)) @ P) @ Sigma^T) @ Q^T + b_tan, c )
pub fn riemann_lowrank_forward(
    x: &ArrayView2<f32>,     // [B, in]
    p: &ArrayView2<f32>,     // [in, r]
    sigma: &ArrayView2<f32>, // [r, r]
    q: &ArrayView2<f32>,     // [out, r]
    b_tan: &ArrayView1<f32>, // [out]
    c: f32,
    epsilon: f32,
) -> Array2<f32> {
    // 1) Project x to ball
    let x_proj = project_to_ball(&x, epsilon);

    // 2) v = Log_0(x_proj)
    let zeros = zeros_like(&x.view());
    let v = poincare_log_at(&zeros.view(), &x_proj.view(), c);

    // 3) low-rank linear in tangent
    // z1 = v @ P  [B, r]
    let z1 = v.dot(p);
    // z2 = z1 @ Sigma^T
    let z2 = z1.dot(&sigma.t());
    // y_tan = z2 @ Q^T + b_tan
    let mut y_tan = z2.dot(&q.t());
    // add tangent bias row-wise
    // add tangent bias row-wise safely
    let b = b_tan.to_owned();
    for mut row in y_tan.axis_iter_mut(Axis(0)) {
        row += &b.view();
    }

    // 4) y = Exp_0(y_tan)
    let zeros_out = Array2::<f32>::zeros((y_tan.nrows(), y_tan.ncols()));
    let y = poincare_exp_at(&zeros_out.view(), &y_tan.view(), c);
    y
}
