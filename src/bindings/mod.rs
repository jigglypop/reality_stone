mod klein;
mod lorentz;
mod mobius;
mod poincare;
mod rbe;
mod spline;

#[macro_use]
mod macros;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Reality Stone - High-performance hyperbolic neural networks in Rust
#[pymodule]
pub fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    // Mobius operations
    mobius::register(m)?;
    // Poincaré ball operations
    poincare::register(m)?;
    // Lorentz operations
    lorentz::register(m)?;
    // Klein operations
    klein::register(m)?;
    // Spline Layer
    spline::register_spline_module(_py, m)?;
    // RBE (Riemannian Basis Encoding) Module
    rbe::init_module(_py, m)?;
    Ok(())
}
