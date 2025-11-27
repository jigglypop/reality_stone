pub mod klein;
pub mod lorentz;
mod metrikey;
mod mobius;
pub mod poincare;
mod riemann;
mod spline;
mod suppression;
pub mod geodesic_attention;
pub mod unified_riemannian;
pub mod diffusion;
mod extraction;
mod rsulf;
mod bellman;

#[macro_use]
mod macros;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Reality Stone - High-performance hyperbolic neural networks in Rust
#[pymodule]
pub fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Mobius operations
    mobius::register(m)?;
    // Poincaré ball operations
    poincare::register(m)?;
    // Lorentz operations
    lorentz::register(m)?;
    // Klein operations
    klein::register(m)?;
    // Riemann low-rank ops
    riemann::register(m)?;
    // Spline Layer
    spline::register_spline_module(_py, m)?;
    // Suppression Field
    suppression::register(m)?;
    // Unified Riemannian Layer
    unified_riemannian::register(m)?;
    // RBE (Riemannian Basis Encoding) Module - disabled
    // rbe::init_module(_py, m)?;
    // MetriKey Module
    metrikey::init_module(_py, m)?;
    // Riemannian Diffusion
    diffusion::register(m)?;
    // Riemannian Metric Extraction (CUDA)
    extraction::register(m)?;
    // RS-ULF Layer
    rsulf::register(m)?;
    // Bellman Geodesic
    bellman::register(m)?;

    {
        use crate::bindings::poincare as pc;
        m.add_function(wrap_pyfunction!(pc::poincare_to_lorentz_cpu, m)?)?;
        m.add_function(wrap_pyfunction!(pc::poincare_to_klein_cpu, m)?)?;
    }
    // Root-level CUDA symbol re-exports expected by tests
    #[cfg(feature = "cuda")]
    {
        use crate::bindings::{klein as kl, lorentz as lo, mobius as mb, poincare as pc};
        m.add_function(wrap_pyfunction!(mb::mobius_add_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(mb::mobius_scalar_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(pc::poincare_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(pc::poincare_ball_layer_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(pc::poincare_ball_layer_backward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lo::lorentz_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lo::lorentz_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lo::lorentz_ball_layer_backward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(kl::klein_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(kl::klein_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(kl::klein_ball_layer_backward_cuda, m)?)?;
    }
    // Geodesic Attention (CUDA)
    geodesic_attention::register(m)?;
    Ok(())
}
