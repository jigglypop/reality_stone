use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

const EPS: f32 = 1e-7;

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
            
            // Minkowski inner product: u0*v0 - u1*v1 - u2*v2 - ...
            *inner = u_row[0] * v_row[0];
            for j in 1..u_row.len() {
                *inner -= u_row[j] * v_row[j];
            }
        });
    
    result
}

pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let inner = lorentz_inner(u, v);
    let sqrtc = c.sqrt();
    
    inner.mapv(|x| (-x).max(1.0 + EPS).acosh() / sqrtc)
}

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
            
            // Compute inner products
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
            let gamma_u = 1.0 / beta_u.sqrt();
            let gamma_v = 1.0 / beta_v.sqrt();
            let gamma_uv = -uv / (c * (beta_u * beta_v).sqrt());
            
            for j in 0..dim {
                row[j] = gamma_uv * (gamma_u * u_row[j] / (1.0 + gamma_u) + 
                                    gamma_v * v_row[j] / (1.0 + gamma_v)) + 
                        u_row[j] + v_row[j];
            }
        });
    
    result
}

pub fn lorentz_scalar(u: &ArrayView2<f32>, _c: f32, r: f32) -> Array2<f32> {
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
            
            let norm = (space_norm_sq / (time_comp * time_comp - 1.0).max(EPS)).sqrt();
            let theta = norm.min(1.0 - EPS).atanh() * r;
            let scale = theta.tanh() / norm.max(EPS);
            
            // Set time component
            let mut scaled_space_norm_sq = 0.0;
            for j in 1..dim {
                row[j] = u_row[j] * scale;
                scaled_space_norm_sq += row[j] * row[j];
            }
            row[0] = (1.0 + scaled_space_norm_sq).sqrt();
        });
    
    result
}

pub fn lorentz_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols() - 1;
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let sqrtc = c.sqrt();
            let x0 = x_row[0] * sqrtc;
            let denom = (x0 + 1.0).max(EPS);
            
            for j in 0..dim {
                row[j] = (x_row[j + 1] * sqrtc) / denom;
            }
        });
    
    result
}

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