//! Typed runtime-facing structs shared by the canonical Clarus compute core.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Wake,
    Nrem,
    Rem,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellState {
    pub activation: f32,
    pub refractory: f32,
    pub memory_trace: f32,
    pub adaptation: f32,
    pub stp_u: f32,
    pub stp_x: f32,
    pub bit: u8,
}

impl Default for CellState {
    fn default() -> Self {
        Self {
            activation: 0.0,
            refractory: 0.0,
            memory_trace: 0.0,
            adaptation: 0.0,
            stp_u: 0.5,
            stp_x: 1.0,
            bit: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxInput {
    /// Packed sparse matrix and state vectors passed into a Rust relax kernel.
    pub values: Vec<f32>,
    pub col_idx: Vec<i32>,
    pub row_ptr: Vec<i32>,
    pub bias: Vec<f32>,
    pub phi: Vec<f32>,
    pub state: Vec<f32>,
    pub mode: Mode,
    pub dt: f32,
    pub max_steps: usize,
    pub tol: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// Metadata only; higher-level snapshot payloads stay in Python for now.
    pub step: usize,
    pub mode: Mode,
    pub active_modules: usize,
    pub energy_budget: usize,
}
