// src/layers/spline.rs

use ndarray::{Array1, Array2};
use ndarray_rand::rand::{thread_rng, Rng};
use numpy::{PyReadonlyArray2, ToPyArray, PyArray2};
use pyo3::prelude::*;
use std::ops::AddAssign;

use crate::ops;

#[pyclass]
pub struct SplineLayer {
    control_points: Array2<f32>,
    pub k: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[pymethods]
impl SplineLayer {
    #[new]
    pub fn new(k: usize, in_features: usize, out_features: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            control_points: Array2::from_shape_fn((k + 1, in_features), |(_, _)| rng.gen::<f32>() * 0.02),
            k,
            in_features,
            out_features,
        }
    }

    #[staticmethod]
    pub fn from_weight_py(
        py: Python,
        weight: PyReadonlyArray2<f32>,
        k: usize,
        learning_rate: f32,
        steps: usize,
    ) -> PyResult<Self> {
        let weight_array = weight.as_array().to_owned();
        Ok(Self::from_weight(&weight_array, k, learning_rate, steps))
    }

    #[getter]
    pub fn get_control_points<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.control_points.to_pyarray(py)
    }

    #[setter]
    pub fn set_control_points(&mut self, control_points: PyReadonlyArray2<f32>) -> PyResult<()> {
        self.control_points = control_points.as_array().to_owned();
        Ok(())
    }

    pub fn forward<'py>(&self, py: Python<'py>, input: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {
        let input_array = input.as_array();
        let weight = self.interpolate_internal();
        let output = input_array.dot(&weight.t());
        output.to_pyarray(py)
    }

    pub fn interpolate<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.interpolate_internal().to_pyarray(py)
    }

    pub fn get_compression_ratio(&self) -> f32 {
        let original_params = (self.in_features * self.out_features) as f32;
        let compressed_params = self.control_points.len() as f32;
        original_params / compressed_params
    }
}

impl SplineLayer {
    pub fn from_weight(weight: &Array2<f32>, k: usize, learning_rate: f32, steps: usize) -> Self {
        let (out_features, in_features) = weight.dim();
        let mut rng = thread_rng();
        let mut control_points = Array2::from_shape_fn((k + 1, in_features), |(_, _)| rng.gen::<f32>() * 0.02);

        for _ in 0..steps {
            let (reconstructed_weight, grad) = Self::interpolate_with_grad(&control_points, out_features);
            let loss_grad = ops::mse_loss_grad(&reconstructed_weight, weight);
            let mut control_points_grad = Array2::<f32>::zeros((k + 1, in_features));
            
            for i in 0..out_features {
                let t = i as f32 / (out_features - 1) as f32;
                let t_scaled = t * k as f32;
                let j = (t_scaled.floor() as usize).clamp(1, k - 2);

                let p0_grad = grad.p0_grads[i].clone() * &loss_grad.row(i);
                let p1_grad = grad.p1_grads[i].clone() * &loss_grad.row(i);
                let p2_grad = grad.p2_grads[i].clone() * &loss_grad.row(i);
                let p3_grad = grad.p3_grads[i].clone() * &loss_grad.row(i);

                control_points_grad.row_mut(j - 1).add_assign(&p0_grad);
                control_points_grad.row_mut(j).add_assign(&p1_grad);
                control_points_grad.row_mut(j + 1).add_assign(&p2_grad);
                control_points_grad.row_mut(j + 2).add_assign(&p3_grad);
            }
            control_points.scaled_add(-learning_rate, &control_points_grad);
        }
        Self { control_points, k, in_features, out_features }
    }

    fn interpolate_internal(&self) -> Array2<f32> {
        let mut reconstructed = Array2::zeros((self.out_features, self.in_features));
        for i in 0..self.out_features {
            let t = i as f32 / (self.out_features - 1) as f32;
            let t_scaled = t * self.k as f32;
            let j = (t_scaled.floor() as usize).clamp(1, self.k - 2);
            let t_local = t_scaled - j as f32;
            let t2 = t_local * t_local;
            let t3 = t2 * t_local;

            let c0 = -0.5 * t3 + t2 - 0.5 * t_local;
            let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local;
            let c3 = 0.5 * t3 - 0.5 * t2;
            
            let p0 = self.control_points.row(j - 1);
            let p1 = self.control_points.row(j);
            let p2 = self.control_points.row(j + 1);
            let p3 = self.control_points.row(j + 2);

            reconstructed.row_mut(i).assign(&(c0 * &p0 + c1 * &p1 + c2 * &p2 + c3 * &p3));
        }
        reconstructed
    }

    fn interpolate_with_grad(control_points: &Array2<f32>, out_features: usize) -> (Array2<f32>, CatmullRomGradients) {
        let mut reconstructed = Array2::zeros((out_features, control_points.shape()[1]));
        let k = control_points.shape()[0] - 1;
        let mut grads = CatmullRomGradients {
            p0_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p1_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p2_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p3_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
        };

        for i in 0..out_features {
            let t = i as f32 / (out_features - 1) as f32;
            let t_scaled = t * k as f32;
            let j = (t_scaled.floor() as usize).clamp(1, k - 2);
            let t_local = t_scaled - j as f32;
            let t2 = t_local * t_local;
            let t3 = t2 * t_local;

            let c0 = -0.5 * t3 + t2 - 0.5 * t_local;
            let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local;
            let c3 = 0.5 * t3 - 0.5 * t2;
            
            let p0 = control_points.row(j - 1);
            let p1 = control_points.row(j);
            let p2 = control_points.row(j + 1);
            let p3 = control_points.row(j + 2);
            
            reconstructed.row_mut(i).assign(&(c0 * &p0 + c1 * &p1 + c2 * &p2 + c3 * &p3));
            
            grads.p0_grads[i].fill(c0);
            grads.p1_grads[i].fill(c1);
            grads.p2_grads[i].fill(c2);
            grads.p3_grads[i].fill(c3);
        }
        (reconstructed, grads)
    }
}

struct CatmullRomGradients {
    p0_grads: Vec<Array1<f32>>,
    p1_grads: Vec<Array1<f32>>,
    p2_grads: Vec<Array1<f32>>,
    p3_grads: Vec<Array1<f32>>,
} 