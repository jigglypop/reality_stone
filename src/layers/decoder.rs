use super::rsulf::{randomized_svd, GlobalBasis};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

pub struct RiemannianDecoder {
    pub d_model: usize,
    pub k: usize,
    pub r: usize,
    pub vocab: usize,
    pub u: Array2<f32>,
    pub a: Array2<f32>,
    pub bt: Array2<f32>,
    pub bias: Array1<f32>,
}

impl RiemannianDecoder {
    pub fn new(u: Array2<f32>, a: Array2<f32>, bt: Array2<f32>, bias: Array1<f32>) -> Self {
        let d_model = u.nrows();
        let k = u.ncols();
        let vocab = bt.nrows();
        let r = bt.ncols();
        Self {
            d_model,
            k,
            r,
            vocab,
            u,
            a,
            bt,
            bias,
        }
    }

    pub fn from_lm_head(
        w_lm: ArrayView2<f32>,
        b_lm: ArrayView1<f32>,
        global_basis: &GlobalBasis,
        target_rank: usize,
    ) -> Self {
        let vocab = w_lm.nrows();
        let d_model = w_lm.ncols();
        let u_basis = global_basis.u.view();
        let k_basis = global_basis.rank.min(d_model);
        let u = u_basis.slice(s![.., 0..k_basis]).to_owned();
        let w_tilde = w_lm.dot(&u);
        let max_rank = target_rank.min(k_basis).min(vocab);
        let (u_svd, s_svd, v_svd) = randomized_svd(&w_tilde, max_rank, 20, 5);
        let r = u_svd.ncols();
        let mut bt = Array2::<f32>::zeros((vocab, r));
        for j in 0..r {
            let s_val = s_svd[j].sqrt();
            for i in 0..vocab {
                bt[[i, j]] = u_svd[[i, j]] * s_val;
            }
        }
        let mut a = Array2::<f32>::zeros((r, k_basis));
        for j in 0..r {
            let s_val = s_svd[j].sqrt();
            for i in 0..k_basis {
                a[[j, i]] = s_val * v_svd[[i, j]];
            }
        }
        let bias = b_lm.to_owned();
        RiemannianDecoder::new(u, a, bt, bias)
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let c = x.dot(&self.u);
        let q = c.dot(&self.a.t());
        let mut logits = q.dot(&self.bt.t());
        for i in 0..self.vocab {
            let b = self.bias[i];
            for j in 0..logits.nrows() {
                logits[[j, i]] += b;
            }
        }
        logits
    }
}
