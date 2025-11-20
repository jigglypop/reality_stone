use ndarray::{Array2, ArrayView2, Zip};

/// Computes the dynamic suppression field epsilon(x) = base + linear * x + hyp * tanh(scale * x)
/// element-wise.
pub fn compute_dynamic_suppression(
    x: &ArrayView2<f32>,
    base: f32,
    linear: f32,
    hyp: f32,
    scale: f32,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((x.nrows(), x.ncols()));
    
    Zip::from(&mut out).and(x).par_for_each(|y, &val| {
        *y = base + linear * val + hyp * (scale * val).tanh();
    });
    
    out
}

