// 동적 곡률 구조체
#[derive(Debug, Clone)]
pub struct DynamicCurvature {
    pub kappa: f32,
    pub c_min: f32,
    pub c_max: f32,
}

impl DynamicCurvature {
    pub fn new(kappa: f32, c_min: f32, c_max: f32) -> Self {
        Self {
            kappa,
            c_min,
            c_max,
        }
    }

    pub fn compute_c(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }

    pub fn compute_dc_dkappa(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}

#[derive(Debug, Clone)]
pub struct LayerWiseDynamicCurvature {
    pub kappas: Vec<f32>,
    pub c_min: f32,
    pub c_max: f32,
}

impl LayerWiseDynamicCurvature {
    pub fn new(num_layers: usize, c_min: f32, c_max: f32) -> Self {
        Self {
            kappas: vec![0.0; num_layers],
            c_min,
            c_max,
        }
    }

    pub fn from_kappas(kappas: Vec<f32>, c_min: f32, c_max: f32) -> Self {
        Self {
            kappas,
            c_min,
            c_max,
        }
    }

    pub fn compute_c(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }

    pub fn compute_dc_dkappa(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dynamic_curvature_limits() {
        let dc_min = DynamicCurvature::new(-20.0, 0.1, 1.0);
        let dc_max = DynamicCurvature::new(20.0, 0.1, 1.0);
        assert_relative_eq!(dc_min.compute_c(), 0.1, epsilon = 1e-3);
        assert_relative_eq!(dc_max.compute_c(), 1.0, epsilon = 1e-3);
    }

    #[test]
    fn dynamic_curvature_derivative_matches_numeric() {
        let kappa = 0.5_f32;
        let c_min = 0.1_f32;
        let c_max = 1.0_f32;
        let base = DynamicCurvature::new(kappa, c_min, c_max);
        let h = 1e-3_f32;
        let plus = DynamicCurvature::new(kappa + h, c_min, c_max);
        let minus = DynamicCurvature::new(kappa - h, c_min, c_max);
        let num = (plus.compute_c() - minus.compute_c()) / (2.0 * h);
        let ana = base.compute_dc_dkappa();
        assert_relative_eq!(num, ana, epsilon = 1e-3);
    }

    #[test]
    fn layerwise_curvature_limits_and_derivative() {
        let mut kappas = Vec::new();
        kappas.push(-20.0);
        kappas.push(0.0);
        kappas.push(20.0);
        let lw = LayerWiseDynamicCurvature::from_kappas(kappas, 0.2, 0.8);
        let c0 = lw.compute_c(0);
        let c2 = lw.compute_c(2);
        assert_relative_eq!(c0, 0.2, epsilon = 1e-3);
        assert_relative_eq!(c2, 0.8, epsilon = 1e-3);
        let h = 1e-3_f32;
        let mut kappas_fd = Vec::new();
        kappas_fd.push(h);
        kappas_fd.push(0.0);
        kappas_fd.push(0.0);
        let lw_plus = LayerWiseDynamicCurvature::from_kappas(kappas_fd.clone(), 0.2, 0.8);
        let mut kappas_fd_minus = Vec::new();
        kappas_fd_minus.push(-h);
        kappas_fd_minus.push(0.0);
        kappas_fd_minus.push(0.0);
        let lw_minus = LayerWiseDynamicCurvature::from_kappas(kappas_fd_minus, 0.2, 0.8);
        let num = (lw_plus.compute_c(0) - lw_minus.compute_c(0)) / (2.0 * h);
        let ana = lw.compute_dc_dkappa(0);
        assert_relative_eq!(num, ana, epsilon = 1e-3);
    }
}
