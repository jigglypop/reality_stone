use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct ControlPoint {
    pub t: usize,
    pub x: Array1<f32>,
    pub v: Array1<f32>,
}

pub struct GeodesicMemory {
    pub d_model: usize,
    pub threshold: f32,
    pub control_points: VecDeque<ControlPoint>,
    buffer: Vec<(usize, Array1<f32>)>,
    last_t: usize,
}

impl GeodesicMemory {
    pub fn new(d_model: usize, threshold: f32) -> Self {
        Self {
            d_model,
            threshold,
            control_points: VecDeque::new(),
            buffer: Vec::new(),
            last_t: 0,
        }
    }

    pub fn push(&mut self, t: usize, x: ArrayView1<f32>) -> bool {
        let x_owned = x.to_owned();
        self.buffer.push((t, x_owned));
        self.last_t = t;

        if self.buffer.len() < 3 {
            if self.control_points.is_empty() {
                self.add_control_point(t);
                return true;
            }
            return false;
        }

        let len = self.buffer.len();
        let (t_curr, x_curr) = &self.buffer[len - 1];
        let (_t_prev, x_prev) = &self.buffer[len - 2];
        let (_t_prev2, x_prev2) = &self.buffer[len - 3];

        let acc = x_curr - x_prev * 2.0 + x_prev2;
        let acc_norm = acc.dot(&acc).sqrt();

        if acc_norm > self.threshold {
            self.add_control_point(*t_curr);
            let last = self.buffer.pop().unwrap();
            self.buffer.clear();
            self.buffer.push(last);
            return true;
        }

        false
    }

    fn add_control_point(&mut self, t: usize) {
        let v = if let Some((_, x_last)) = self.buffer.last() {
            if self.buffer.len() >= 2 {
                let x_prev = &self.buffer[self.buffer.len() - 2].1;
                x_last - x_prev
            } else {
                Array1::zeros(self.d_model)
            }
        } else {
            Array1::zeros(self.d_model)
        };

        let x = if let Some((_, x_val)) = self.buffer.last() {
            x_val.clone()
        } else {
            Array1::zeros(self.d_model)
        };

        self.control_points.push_back(ControlPoint { t, x, v });
    }

    pub fn query(&self, t: f32) -> Array1<f32> {
        if self.control_points.is_empty() {
            return Array1::zeros(self.d_model);
        }

        let idx = self
            .control_points
            .binary_search_by(|cp| cp.t.partial_cmp(&(t as usize)).unwrap())
            .unwrap_or_else(|x| x);

        if idx == 0 {
            if self.control_points.len() == 1 && !self.buffer.is_empty() {
                if let Some((t_buf, _x_buf)) = self.buffer.last() {
                    if t > (self.control_points[0].t as f32) && t <= (*t_buf as f32) {}
                }
            }
            return self.control_points[0].x.clone();
        }
        if idx >= self.control_points.len() {
            if !self.buffer.is_empty() {
                let (t_last, x_last) = self.buffer.last().unwrap();
                let cp_last = self.control_points.back().unwrap();
                if t > (cp_last.t as f32) && t <= (*t_last as f32) {
                    let p0 = cp_last;
                    let v_tip = if self.buffer.len() >= 2 {
                        &self.buffer[self.buffer.len() - 1].1
                            - &self.buffer[self.buffer.len() - 2].1
                    } else {
                        Array1::zeros(self.d_model)
                    };

                    let t0 = p0.t as f32;
                    let t1 = *t_last as f32;
                    let s = (t - t0) / (t1 - t0);
                    let s2 = s * s;
                    let s3 = s2 * s;

                    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                    let h10 = s3 - 2.0 * s2 + s;
                    let h01 = -2.0 * s3 + 3.0 * s2;
                    let h11 = s3 - s2;

                    let dt = t1 - t0;
                    let m0 = &p0.v * dt;
                    let m1 = &v_tip * dt;
                    return &p0.x * h00 + &m0 * h10 + x_last * h01 + &m1 * h11;
                }
            }
            return self.control_points.back().unwrap().x.clone();
        }

        let p0 = &self.control_points[idx - 1];
        let p1 = &self.control_points[idx];

        let t0 = p0.t as f32;
        let t1 = p1.t as f32;

        if t1 == t0 {
            return p0.x.clone();
        }

        let s = (t - t0) / (t1 - t0);
        let s2 = s * s;
        let s3 = s2 * s;

        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;

        let dt = t1 - t0;
        let m0 = &p0.v * dt;
        let m1 = &p1.v * dt;

        let x_t = &p0.x * h00 + &m0 * h10 + &p1.x * h01 + &m1 * h11;

        x_t
    }

    pub fn get_compression_stats(&self) -> (usize, usize, f32) {
        let stored = self.control_points.len();
        let covered = if self.control_points.is_empty() {
            0
        } else {
            self.last_t + 1
        };
        let ratio = if stored > 0 {
            covered as f32 / stored as f32
        } else {
            0.0
        };
        (stored, covered, ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_spline_compression_sine_wave() {
        let seq_len = 100;
        let d_model = 16;
        let mut trajectory = Vec::new();

        for t in 0..seq_len {
            let mut state = Array1::<f32>::zeros(d_model);
            for i in 0..d_model {
                state[i] = ((t as f32) * 0.1 + (i as f32)).sin();
            }
            trajectory.push(state);
        }

        let mut memory = GeodesicMemory::new(d_model, 0.01);

        for (t, state) in trajectory.iter().enumerate() {
            memory.push(t, state.view());
        }

        let (stored, covered, ratio) = memory.get_compression_stats();
        println!(
            "Compression: Stored {} / Covered {} (Ratio {:.2}x)",
            stored, covered, ratio
        );

        assert!(stored < seq_len, "Should compress somewhat");
        assert!(stored > 2, "Should have more than start/end points");

        let mut total_mse = 0.0;
        let mut max_mse = 0.0;

        for (t, original) in trajectory.iter().enumerate() {
            let reconstructed = memory.query(t as f32);
            let diff = original - &reconstructed;
            let mse = diff.dot(&diff) / d_model as f32;

            total_mse += mse;
            if mse > max_mse {
                max_mse = mse;
            }
        }
        let avg_mse = total_mse / seq_len as f32;
        println!(
            "Reconstruction Error: Avg MSE {:.6}, Max MSE {:.6}",
            avg_mse, max_mse
        );

        assert!(avg_mse < 0.01, "Average reconstruction error too high");
    }

    #[test]
    fn test_linear_trajectory_compression() {
        let seq_len = 50;
        let d_model = 4;
        let mut trajectory = Vec::new();

        for t in 0..seq_len {
            let mut state = Array1::<f32>::zeros(d_model);
            for i in 0..d_model {
                state[i] = (t as f32) * 0.1;
            }
            trajectory.push(state);
        }

        let mut memory = GeodesicMemory::new(d_model, 0.001);

        for (t, state) in trajectory.iter().enumerate() {
            memory.push(t, state.view());
        }

        let (stored, _, _) = memory.get_compression_stats();
        println!("Linear Compression: Stored {}", stored);
    }
}
