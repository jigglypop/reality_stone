pub mod memory;
pub mod spline;
pub mod spline_cache;
pub mod rsulf;

pub mod bellman;
pub mod diffusion;
pub mod extraction;
pub mod geodesic_attention;
pub mod klein;
pub mod lorentz;
pub mod macros;
pub mod metrikey;
pub mod mobius;
pub mod poincare;
pub mod riemann;
pub mod suppression;
pub mod unified_riemannian;

use pyo3::prelude::*;

#[pymodule]
pub fn _rust(py: Python, m: &PyModule) -> PyResult<()> {
    rsulf::register(m)?;
    spline::register_spline_module(py, m)?;
    m.add_class::<spline_cache::PySplineCache>()?;
    m.add_class::<memory::PyGeodesicMemory>()?;
    
    Ok(())
}
