use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct TinyMLP {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl TinyMLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            w1: Array2::zeros((input_dim, hidden_dim)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::zeros((hidden_dim, output_dim)),
            b2: Array1::zeros(output_dim),
        }
    }

    pub fn from_weights(
        w1: Array2<f32>,
        b1: Array1<f32>,
        w2: Array2<f32>,
        b2: Array1<f32>,
    ) -> Self {
        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let h = x.dot(&self.w1) + &self.b1;
        let h_act = h.mapv(|v| v.max(0.0));

        h_act.dot(&self.w2) + &self.b2
    }
}

#[derive(Debug, Clone)]
pub struct HyperMetric {
    pub u_global: Array2<f32>,
    pub v_global: Array2<f32>,
    pub hypernet: TinyMLP,
    pub r: usize,
    pub d_model: usize,
}

impl HyperMetric {
    pub fn new(d_model: usize, r: usize, hyper_hidden: usize) -> Self {
        let hyper_input_dim = 64;
        let output_dim = r * r;

        Self {
            u_global: Array2::zeros((d_model, r)),
            v_global: Array2::zeros((d_model, r)),
            hypernet: TinyMLP::new(hyper_input_dim, hyper_hidden, output_dim),
            r,
            d_model,
        }
    }

    pub fn from_components(
        u_global: Array2<f32>,
        v_global: Array2<f32>,
        hypernet: TinyMLP,
    ) -> Self {
        let d_model = u_global.nrows();
        let r = u_global.ncols();
        Self {
            u_global,
            v_global,
            hypernet,
            r,
            d_model,
        }
    }

    pub fn generate_core(&self, layer_emb: &Array1<f32>) -> Array2<f32> {
        let flat_core = self.hypernet.forward(layer_emb);
        flat_core.into_shape((self.r, self.r)).unwrap().to_owned()
    }

    pub fn project_forward(&self, x: &Array2<f32>, layer_emb: &Array1<f32>) -> Array2<f32> {
        let core = self.generate_core(layer_emb);

        let x_proj = x.dot(&self.u_global);
        let x_core = x_proj.dot(&core);

        x_core.dot(&self.v_global.t())
    }
}
