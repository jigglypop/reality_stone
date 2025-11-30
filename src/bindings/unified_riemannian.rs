// ============================================================================
// 파일: src/bindings/unified_riemannian.rs
// 목적: 통합 리만 레이어 Python 바인딩
// ============================================================================

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use crate::layers::unified_riemannian::*;

#[pyclass]
pub struct PyUnifiedRiemannianLayer {
    inner: UnifiedRiemannianLayer,
}

#[pymethods]
impl PyUnifiedRiemannianLayer {
    #[new]
    #[pyo3(signature = (metric_type, curvature=1.0, input_dim=64, enable_bellman=false, gamma=0.99))]
    fn new(
        metric_type: &str,
        curvature: f32,
        input_dim: usize,
        enable_bellman: bool,
        gamma: f32,
    ) -> PyResult<Self> {
        let mut layer = UnifiedRiemannianLayer::new(
            metric_type,
            curvature,
            input_dim,
            enable_bellman,
        );
        layer.lagrangian_params.gamma = gamma;
        Ok(Self { inner: layer })
    }
    
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        target: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<(&'py PyArray2<f32>, Option<PyObject>)> {
        let x_arr = x.as_array();
        let target_arr = target.as_ref().map(|t| t.as_array());
        
        let output = self.inner.forward(&x_arr, target_arr.as_ref());
        
        let output_py = output.output.into_pyarray(py);
        let energy_py = output.energy.map(|e| {
            let dict = PyDict::new(py);
            dict.set_item("kinetic", e.kinetic.into_pyarray(py)).unwrap();
            dict.set_item("potential", e.potential.into_pyarray(py)).unwrap();
            dict.set_item("lagrangian", e.lagrangian.into_pyarray(py)).unwrap();
            dict.set_item("bellman_residual", e.bellman_residual.into_pyarray(py)).unwrap();
            dict.into()
        });
        
        Ok((output_py, energy_py))
    }
    
    fn backward<'py>(
        &self,
        py: Python<'py>,
        grad_output: PyReadonlyArray2<f32>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let grad_arr = grad_output.as_array();
        let x_arr = x.as_array();
        
        // 더미 캐시 생성
        let cache = crate::layers::unified_riemannian::LayerCache {
            input: x_arr.to_owned(),
            velocity: None,
            metric_values: ndarray::Array2::zeros((x_arr.nrows(), x_arr.ncols())),
        };
        
        let grads = self.inner.backward(&grad_arr, &x_arr, &cache);
        Ok(grads.grad_input.into_pyarray(py))
    }
    
    fn geodesic_path<'py>(
        &self,
        py: Python<'py>,
        start: PyReadonlyArray2<f32>,
        end: PyReadonlyArray2<f32>,
        num_steps: usize,
    ) -> PyResult<Vec<&'py PyArray2<f32>>> {
        let start_arr = start.as_array();
        let end_arr = end.as_array();
        
        let path = self.inner.geodesic_path(&start_arr, &end_arr, num_steps);
        
        Ok(path.into_iter()
            .map(|p| p.into_pyarray(py))
            .collect())
    }
    
    fn compute_energy<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        v: PyReadonlyArray2<f32>,
        x_next: PyReadonlyArray2<f32>,
        reward: PyReadonlyArray1<f32>,
    ) -> PyResult<PyObject> {
        let energy = self.inner.compute_energy(
            &x.as_array(),
            &v.as_array(),
            &x_next.as_array(),
            &reward.as_array(),
        );
        
        let dict = PyDict::new(py);
        dict.set_item("kinetic", energy.kinetic.into_pyarray(py))?;
        dict.set_item("potential", energy.potential.into_pyarray(py))?;
        dict.set_item("lagrangian", energy.lagrangian.into_pyarray(py))?;
        dict.set_item("bellman_residual", energy.bellman_residual.into_pyarray(py))?;
        Ok(dict.into())
    }
    
    fn flow_step<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        num_steps: usize,
        learning_rate: f32,
    ) -> PyResult<&'py PyArray2<f32>> {
        let result = self.inner.flow_step(
            &x.as_array(),
            num_steps,
            learning_rate,
        );
        Ok(result.into_pyarray(py))
    }
    
    fn update_value_function(
        &mut self,
        x: PyReadonlyArray2<f32>,
        x_next: PyReadonlyArray2<f32>,
        reward: PyReadonlyArray1<f32>,
        learning_rate: f32,
    ) -> PyResult<()> {
        self.inner.update_value_function(
            &x.as_array(),
            &x_next.as_array(),
            &reward.as_array(),
            learning_rate,
        );
        Ok(())
    }
    
    fn update_metric(
        &mut self,
        x: PyReadonlyArray2<f32>,
        v: PyReadonlyArray2<f32>,
        learning_rate: f32,
    ) -> PyResult<()> {
        self.inner.update_metric(
            &x.as_array(),
            &v.as_array(),
            learning_rate,
        );
        Ok(())
    }
}

// 독립 함수들
#[pyfunction]
pub fn compute_metric<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
) -> PyResult<&'py PyArray2<f32>> {
    use crate::layers::metric::*;
    
    let metric: Box<dyn MetricTensor> = match metric_type {
        "poincare" => Box::new(PoincareMetric::new(curvature)),
        "lorentz" => Box::new(LorentzMetric::new(curvature)),
        "klein" => Box::new(KleinMetric::new(curvature)),
        "diagonal" => Box::new(DiagonalMetric::new(x.shape()[1])),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown metric type: {}", metric_type)
        )),
    };
    
    let x_arr = x.as_array();
    let result = metric.compute_metric(&x_arr);
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn geodesic_distance<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
) -> PyResult<&'py PyArray1<f32>> {
    use crate::layers::metric::*;
    
    let metric_enum = match metric_type {
        "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
        "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
        "klein" => MetricType::Klein(KleinMetric::new(curvature)),
        "diagonal" => MetricType::Diagonal(DiagonalMetric::new(x.shape()[1])),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown metric type: {}", metric_type)
        )),
    };
    
    let distance = crate::layers::geodesic::geodesic_distance(
        &metric_enum,
        &x.as_array(),
        &y.as_array(),
    );
    Ok(distance.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (x, y, metric_type, curvature, t=0.5))]
pub fn geodesic_interpolate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    metric_type: &str,
    curvature: f32,
    t: f32,
) -> PyResult<&'py PyArray2<f32>> {
    use crate::layers::metric::*;
    
    let metric_enum = match metric_type {
        "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
        "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
        "klein" => MetricType::Klein(KleinMetric::new(curvature)),
        "diagonal" => MetricType::Diagonal(DiagonalMetric::new(x.shape()[1])),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown metric type: {}", metric_type)
        )),
    };
    
    let result = crate::layers::geodesic::geodesic_interpolation(
        &metric_enum,
        &x.as_array(),
        &y.as_array(),
        t,
    );
    Ok(result.into_pyarray(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyUnifiedRiemannianLayer>()?;
    m.add_function(wrap_pyfunction!(compute_metric, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_distance, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_interpolate, m)?)?;
    Ok(())
}

