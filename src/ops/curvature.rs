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
