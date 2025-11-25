use ndarray::{Array2, ArrayView2};

#[cfg(feature = "cuda")]
extern "C" {
    fn fast_extract_metric_cuda(
        W: *const f32,
        U: *mut f32,
        G: *mut f32,
        V: *mut f32,
        out_dim: i32,
        in_dim: i32,
        k: i32,
    );
}

/// CUDA 기반 리만 메트릭 추출 (Fast Random Projection)
#[cfg(feature = "cuda")]
pub fn extract_metric_cuda(
    w: ArrayView2<f32>,
    _calibration_data: ArrayView2<f32>,
    target_dim: usize,
    _num_steps: usize,
    _curvature: f32,
    _lr: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let out_dim = w.nrows();
    let in_dim = w.ncols();
    let k = target_dim;

    let mut u = Array2::<f32>::zeros((out_dim, k));
    let mut g = Array2::<f32>::zeros((k, k));
    let mut v = Array2::<f32>::zeros((in_dim, k));

    unsafe {
        fast_extract_metric_cuda(
            w.as_ptr(),
            u.as_mut_ptr(),
            g.as_mut_ptr(),
            v.as_mut_ptr(),
            out_dim as i32,
            in_dim as i32,
            k as i32,
        );
    }

    (u, g, v)
}

/// CPU 폴백 (CUDA 없을 때) - Random Projection
#[cfg(not(feature = "cuda"))]
pub fn extract_metric_cuda(
    w: ArrayView2<f32>,
    _calibration_data: ArrayView2<f32>,
    target_dim: usize,
    _num_steps: usize,
    _curvature: f32,
    _lr: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use rand::Rng;
    
    let out_dim = w.nrows();
    let in_dim = w.ncols();
    let k = target_dim;
    let scale = 1.0 / (k as f32).sqrt();

    let mut u = Array2::<f32>::zeros((out_dim, k));
    let mut g = Array2::<f32>::zeros((k, k));
    let mut v = Array2::<f32>::zeros((in_dim, k));

    let mut rng = rand::thread_rng();
    
    for i in 0..out_dim {
        for j in 0..k {
            u[[i, j]] = rng.gen::<f32>() * scale;
        }
    }
    for i in 0..in_dim {
        for j in 0..k {
            v[[i, j]] = rng.gen::<f32>() * scale;
        }
    }

    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f32;
            for a in 0..out_dim {
                for b in 0..in_dim {
                    sum += u[[a, i]] * w[[a, b]] * v[[b, j]];
                }
            }
            g[[i, j]] = sum;
        }
    }

    (u, g, v)
}

