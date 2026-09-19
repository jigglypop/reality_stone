#[derive(Clone, Copy, Debug)]
pub struct LlmPreEqWeights {
    pub contradicted: f64,
    pub unsupported: f64,
    pub no_evidence: f64,
    pub coverage: f64,
    pub instruction: f64,
    pub self_contradiction: f64,
    pub uncertainty: f64,
}

impl LlmPreEqWeights {
    pub fn validate(&self) -> Result<(), String> {
        let values = [
            self.contradicted,
            self.unsupported,
            self.no_evidence,
            self.coverage,
            self.instruction,
            self.self_contradiction,
            self.uncertainty,
        ];
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("weights must be finite and non-negative".to_string());
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn defect_energies(
    supported: &[f64],
    unsupported: &[f64],
    contradicted: &[f64],
    instruction: &[f64],
    self_contradiction: &[f64],
    uncertainty: &[f64],
    weights: LlmPreEqWeights,
) -> Result<Vec<f64>, String> {
    weights.validate()?;
    let n = supported.len();
    let arrays = [
        unsupported,
        contradicted,
        instruction,
        self_contradiction,
        uncertainty,
    ];
    if arrays.iter().any(|arr| arr.len() != n) {
        return Err("all defect arrays must have the same length".to_string());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let vals = [
            supported[i],
            unsupported[i],
            contradicted[i],
            instruction[i],
            self_contradiction[i],
            uncertainty[i],
        ];
        if vals.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("defect counts must be finite and non-negative".to_string());
        }
        let total_claims = supported[i] + unsupported[i] + contradicted[i];
        let denom = total_claims.max(1.0);
        let no_evidence = if total_claims <= 0.0 {
            weights.no_evidence
        } else {
            0.0
        };
        let energy = weights.contradicted * contradicted[i] / denom
            + weights.unsupported * unsupported[i] / denom
            + no_evidence
            + weights.coverage / (1.0 + supported[i])
            + weights.instruction * instruction[i]
            + weights.self_contradiction * self_contradiction[i]
            + weights.uncertainty * uncertainty[i];
        out.push(energy.max(0.0));
    }
    Ok(out)
}

pub fn gibbs_posterior(prior: &[f64], energy: &[f64], beta: f64) -> Result<Vec<f64>, String> {
    if prior.is_empty() {
        return Err("prior must be non-empty".to_string());
    }
    if prior.len() != energy.len() {
        return Err("prior and energy must have the same length".to_string());
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err("beta must be finite and non-negative".to_string());
    }
    if prior.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("prior must be finite and non-negative".to_string());
    }
    if energy.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("energy must be finite and non-negative".to_string());
    }
    let prior_sum: f64 = prior.iter().sum();
    if prior_sum <= 0.0 {
        return Err("prior must have positive total mass".to_string());
    }
    let mut min_energy = f64::INFINITY;
    for (&p, &e) in prior.iter().zip(energy.iter()) {
        if p > 0.0 && e < min_energy {
            min_energy = e;
        }
    }
    if !min_energy.is_finite() {
        return Err("at least one supported candidate is required".to_string());
    }
    let mut weights = Vec::with_capacity(prior.len());
    let mut total = 0.0;
    for (&p, &e) in prior.iter().zip(energy.iter()) {
        let value = if p > 0.0 {
            (p / prior_sum) * (-beta * (e - min_energy)).exp()
        } else {
            0.0
        };
        weights.push(value);
        total += value;
    }
    if total <= 0.0 || !total.is_finite() {
        return Err("partition function is not positive and finite".to_string());
    }
    Ok(weights.into_iter().map(|v| v / total).collect())
}

#[derive(Clone, Copy, Debug)]
pub struct ClaimActionWeights {
    pub residual: f64,
    pub graph: f64,
    pub tau: f64,
    pub source: f64,
    pub independence: f64,
    pub missing: f64,
    pub instruction: f64,
    pub schema: f64,
    pub coverage: f64,
    pub unsupported: f64,
    pub ce_penalty: f64,
}

impl ClaimActionWeights {
    pub fn validate(&self) -> Result<(), String> {
        let values = [
            self.residual,
            self.graph,
            self.tau,
            self.source,
            self.independence,
            self.missing,
            self.instruction,
            self.schema,
            self.coverage,
            self.unsupported,
            self.ce_penalty,
        ];
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("claim action weights must be finite and non-negative".to_string());
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn claim_answer_actions(
    residual: &[f64],
    graph: &[f64],
    tau: &[f64],
    source_unreliability: &[f64],
    independence: &[f64],
    missing: &[f64],
    instruction: &[f64],
    schema: &[f64],
    coverage: &[f64],
    unsupported: &[f64],
    ce_penalty: &[f64],
    weights: ClaimActionWeights,
) -> Result<Vec<f64>, String> {
    weights.validate()?;
    let n = residual.len();
    let arrays = [
        graph,
        tau,
        source_unreliability,
        independence,
        missing,
        instruction,
        schema,
        coverage,
        unsupported,
        ce_penalty,
    ];
    if arrays.iter().any(|arr| arr.len() != n) {
        return Err("all claim action arrays must have the same length".to_string());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let vals = [
            residual[i],
            graph[i],
            tau[i],
            source_unreliability[i],
            independence[i],
            missing[i],
            instruction[i],
            schema[i],
            coverage[i],
            unsupported[i],
            ce_penalty[i],
        ];
        if vals.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("claim action components must be finite and non-negative".to_string());
        }
        out.push(
            weights.residual * residual[i]
                + weights.graph * graph[i]
                + weights.tau * tau[i]
                + weights.source * source_unreliability[i]
                + weights.independence * independence[i]
                + weights.missing * missing[i]
                + weights.instruction * instruction[i]
                + weights.schema * schema[i]
                + weights.coverage * coverage[i]
                + weights.unsupported * unsupported[i]
                + weights.ce_penalty * ce_penalty[i],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_weights() -> LlmPreEqWeights {
        LlmPreEqWeights {
            contradicted: 3.0,
            unsupported: 1.0,
            no_evidence: 1.0,
            coverage: 0.2,
            instruction: 2.0,
            self_contradiction: 2.0,
            uncertainty: 0.25,
        }
    }

    #[test]
    fn posterior_recovers_prior_at_beta_zero() {
        let out = gibbs_posterior(&[2.0, 1.0], &[10.0, 0.0], 0.0).unwrap();
        assert!((out[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((out[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn energy_is_non_negative_and_penalizes_no_evidence() {
        let e = defect_energies(
            &[0.0, 4.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            default_weights(),
        )
        .unwrap();
        assert!(e[0] > e[1]);
        assert!(e.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn claim_answer_actions_are_weighted_non_negative_components() {
        let weights = ClaimActionWeights {
            residual: 0.1,
            graph: 1.2,
            tau: 0.1,
            source: 1.5,
            independence: 0.8,
            missing: 1.0,
            instruction: 1.0,
            schema: 1.0,
            coverage: 1.0,
            unsupported: 1.0,
            ce_penalty: 1.0,
        };
        let actions = claim_answer_actions(
            &[2.0],
            &[0.5],
            &[0.25],
            &[0.04],
            &[0.25],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            weights,
        )
        .unwrap();
        assert!((actions[0] - 1.085).abs() < 1e-12);
    }
}
