use ndarray::{Array2, ArrayView2, Zip};

/// Computes the Geodesic update for a batch of vectors using a diagonal metric approximation.
///
/// This implements the "Cheat Sheet" derivation:
/// 1. Metric w = Sigmoid(x) (representing importance/value density)
/// 2. Christoffel Gamma = 0.5 * (1/w) * dw/dx
/// 3. Force = -Gamma * velocity^2
/// 4. x_new = x + Force * dt
///
/// # Arguments
///
/// * `input` - Input features (Batch Size x Hidden Dim)
/// * `dt` - Time step for the geodesic flow (learning rate factor)
pub fn compute_diagonal_geodesic_update(
    input: &ArrayView2<f64>,
    dt: f64,
) -> Array2<f64> {
    let mut output = Array2::zeros(input.raw_dim());
    
    // velocity is assumed to be 1.0 (momentum unit) for this simplified flow
    let velocity_sq = 1.0;

    Zip::from(output.rows_mut())
        .and(input.rows())
        .par_for_each(|mut out_row, in_row| {
            for (i, &val) in in_row.iter().enumerate() {
                // 1. Metric Definition: w(x) = Sigmoid(x)
                // To avoid division by zero, we add epsilon or ensure sigmoid range is safe.
                // Sigmoid is naturally (0, 1), so it's safe for division if we don't hit exact 0.
                let w = 1.0 / (1.0 + (-val).exp());
                // 3. Christoffel Symbol (Diagonal): Gamma = 1/2 * (1/w) * dw
                // Gamma = 0.5 * (1/w) * (w * (1-w)) = 0.5 * (1 - w)
                // This simplification works specifically for Sigmoid metric.
                let gamma = 0.5 * (1.0 - w);
                
                // 4. Geodesic Force
                let force = -gamma * velocity_sq;
                
                // 5. Update
                out_row[i] = val + force * dt;
            }
        });

    output
}

/// Inverse computation for backpropagation (Simplified)
/// 
/// For a full layer, we would need the Jacobian of the update function.
/// Given x_new = x + F(x)*dt, dx_new/dx = 1 + F'(x)*dt
pub fn compute_diagonal_geodesic_backward(
    grad_output: &ArrayView2<f64>,
    input: &ArrayView2<f64>,
    dt: f64,
) -> Array2<f64> {
    let mut grad_input = Array2::zeros(input.raw_dim());
    let velocity_sq = 1.0;

    Zip::from(grad_input.rows_mut())
        .and(grad_output.rows())
        .and(input.rows())
        .par_for_each(|mut gin_row, gout_row, in_row| {
            for (i, &val) in in_row.iter().enumerate() {
                let w = 1.0 / (1.0 + (-val).exp());
                
                // F(x) = -0.5 * (1 - w) * v^2
                // F'(x) = -0.5 * (-dw/dx) * v^2 = 0.5 * w(1-w) * v^2
                
                let dw = w * (1.0 - w);
                let d_force = 0.5 * dw * velocity_sq;
                
                // Chain rule: dL/dx = dL/dy * dy/dx
                // dy/dx = 1 + F'(x) * dt
                let dy_dx = 1.0 + d_force * dt;
                
                gin_row[i] = gout_row[i] * dy_dx;
            }
        });

    grad_input
}

