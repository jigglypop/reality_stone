
use ndarray::{Array1, Array2, ArrayView1};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct ControlPoint {
    pub time: f32,
    pub state: Array1<f32>,
    pub velocity: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct SplineCache {
    pub control_points: Vec<ControlPoint>,
    pub curvature: f32,
    pub dimension: usize,
}

impl SplineCache {
    pub fn new(curvature: f32, dimension: usize) -> Self {
        Self {
            control_points: Vec::new(),
            curvature,
            dimension,
        }
    }

    pub fn add_point(&mut self, time: f32, state: ArrayView1<f32>, velocity: ArrayView1<f32>) {
        if state.len() != self.dimension || velocity.len() != self.dimension {
            panic!("Dimension mismatch in SplineCache");
        }
        
        // Ensure time is increasing
        if let Some(last) = self.control_points.last() {
            if time <= last.time {
                // In a real scenario we might update, but for now append only
                return; 
            }
        }

        self.control_points.push(ControlPoint {
            time,
            state: state.to_owned(),
            velocity: velocity.to_owned(),
        });
    }

    pub fn reconstruct(&self, t: f32) -> Option<Array1<f32>> {
        if self.control_points.is_empty() {
            return None;
        }

        // 1. Find interval [t_k, t_{k+1}]
        // Binary search for the interval
        let idx = match self.control_points.binary_search_by(|cp| {
            if cp.time <= t {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(i) => i, // exact match (unlikely with float, but logic holds)
            Err(i) => i, // insertion point
        };

        // idx is where t would be inserted. 
        // So t is between idx-1 and idx.
        
        if idx == 0 {
            // Before first point
            return Some(self.control_points[0].state.clone());
        }
        if idx >= self.control_points.len() {
            // After last point
            return Some(self.control_points.last().unwrap().state.clone());
        }

        let p0_idx = idx - 1;
        let p1_idx = idx;

        let p0 = &self.control_points[p0_idx];
        let p1 = &self.control_points[p1_idx];

        let dt = p1.time - p0.time;
        if dt < 1e-6 {
            return Some(p0.state.clone());
        }

        // Normalized time u in [0, 1]
        let u = (t - p0.time) / dt;
        
        // 2. Cubic Hermite Spline
        // h00 = 2u^3 - 3u^2 + 1
        // h10 = u^3 - 2u^2 + u
        // h01 = -2u^3 + 3u^2
        // h11 = u^3 - u^2
        
        let u2 = u * u;
        let u3 = u2 * u;

        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;

        // Tangents need to be scaled by interval duration dt
        // m0 = v0 * dt
        // m1 = v1 * dt
        let m0 = &p0.velocity * dt;
        let m1 = &p1.velocity * dt;

        let mut interpolated = &p0.state * h00 
                             + &m0 * h10 
                             + &p1.state * h01 
                             + &m1 * h11;

        // 3. Curvature Correction
        // Blueprint: "Correct path using curvature kappa"
        // A simple approximation for negative curvature (hyperbolic) is to "push out" or "pull in"
        // based on the deviation from geodesic.
        // For now, let's implement a placeholder correction that scales with u(1-u) (max at midpoint)
        if self.curvature.abs() > 1e-6 {
             let mid_correction = u * (1.0 - u) * self.curvature;
             // Apply correction in direction of interpolation? 
             // Or simply scale amplitude? 
             // Blueprint 3.1 mentions "Christoffel symbols correction: -0.5 * Gamma * v * v"
             // Here, let's assume a simple radial correction factor.
             // x_corrected = x * (1 + correction)
             interpolated.mapv_inplace(|x| x * (1.0 + mid_correction));
        }

        Some(interpolated)
    }

    pub fn batch_reconstruct(&self, timestamps: ArrayView1<f32>) -> Array2<f32> {
        let n = timestamps.len();
        let mut output = Array2::zeros((n, self.dimension));
        
        for (i, &t) in timestamps.iter().enumerate() {
            if let Some(state) = self.reconstruct(t) {
                output.row_mut(i).assign(&state);
            }
        }
        output
    }
    
    pub fn clear(&mut self) {
        self.control_points.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_spline_reconstruction() {
        let dim = 2;
        let mut cache = SplineCache::new(0.0, dim);
        
        // p0 at t=0: [0, 0], v=[1, 1]
        cache.add_point(0.0, arr1(&[0.0, 0.0]).view(), arr1(&[1.0, 1.0]).view());
        // p1 at t=1: [1, 1], v=[1, 1]
        cache.add_point(1.0, arr1(&[1.0, 1.0]).view(), arr1(&[1.0, 1.0]).view());
        
        // Midpoint t=0.5
        // Linear would be [0.5, 0.5]
        // Cubic with constant velocity should also be close to linear if velocity matches
        
        let res = cache.reconstruct(0.5).unwrap();
        println!("Reconstructed at 0.5: {:?}", res);
        
        // Check roughly
        assert!((res[0] - 0.5).abs() < 0.1);
        assert!((res[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_curvature_effect() {
        let dim = 1;
        let mut cache = SplineCache::new(1.0, dim); // High curvature
        
        cache.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
        cache.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());
        
        let res_curved = cache.reconstruct(0.5).unwrap();
        
        let mut cache_flat = SplineCache::new(0.0, dim);
        cache_flat.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
        cache_flat.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());
        let res_flat = cache_flat.reconstruct(0.5).unwrap();
        
        // Curvature 1.0 adds u(1-u)*k = 0.25 * 1 = 0.25 factor roughly
        // Expect curved result to be larger (or different)
        assert!((res_curved[0] - res_flat[0]).abs() > 0.001);
    }
}

