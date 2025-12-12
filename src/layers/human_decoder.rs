use crate::layers::poincare::poincare_distance;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

const EPS: f32 = 1e-8;

#[derive(Clone, Copy)]
pub struct StageWeights {
    pub logit: f32,
    pub cosine: f32,
    pub geodesic: f32,
}

pub struct HumanStyleDecoder {
    embeddings: Array2<f32>,
    norms: Array1<f32>,
    skeleton: Vec<usize>,
    relation: Vec<usize>,
    object: Vec<usize>,
    relation_weights: StageWeights,
    object_weights: StageWeights,
    curvature: f32,
}

impl HumanStyleDecoder {
    pub fn new(
        embeddings: Array2<f32>,
        skeleton: Vec<usize>,
        relation: Vec<usize>,
        object: Vec<usize>,
        relation_weights: StageWeights,
        object_weights: StageWeights,
        curvature: f32,
    ) -> Self {
        let norm_vec = embeddings
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt().max(EPS))
            .collect::<Vec<f32>>();
        let norms = Array1::from_vec(norm_vec);
        Self {
            embeddings,
            norms,
            skeleton,
            relation,
            object,
            relation_weights,
            object_weights,
            curvature,
        }
    }

    fn masked_argmax(&self, logits: &ArrayView1<f32>, pool: &[usize]) -> Option<usize> {
        if pool.is_empty() {
            return None;
        }
        let mut best_idx = pool[0];
        let mut best_val = logits[best_idx];
        for &idx in pool.iter().skip(1) {
            let val = logits[idx];
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    fn select_topk(&self, logits: &ArrayView1<f32>, pool: &[usize], k: usize) -> Vec<usize> {
        if pool.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut pairs = pool
            .iter()
            .map(|idx| (*idx, logits[*idx]))
            .collect::<Vec<(usize, f32)>>();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k.min(pairs.len()));
        pairs.into_iter().map(|(idx, _)| idx).collect()
    }

    fn cosine_with_context(&self, idx: usize, context: &ArrayView1<f32>, ctx_norm: f32) -> f32 {
        let embed = self.embeddings.row(idx);
        let dot = embed.dot(context);
        dot / (self.norms[idx] * ctx_norm.max(EPS))
    }

    fn poincare_distance_single(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {
        if self.curvature <= 0.0 {
            return self.euclidean_distance(idx, context);
        }
        let embed = self.embeddings.slice(s![idx..idx + 1, ..]);
        let dist = poincare_distance(&embed.view(), context, self.curvature, EPS);
        dist[0]
    }

    fn euclidean_distance(&self, idx: usize, context: &ArrayView2<f32>) -> f32 {
        let embed = self.embeddings.row(idx);
        let ctx = context.row(0);
        let diff = &embed - &ctx;
        diff.dot(&diff).sqrt()
    }

    fn select_relation(
        &self,
        logits: &ArrayView1<f32>,
        context: &ArrayView1<f32>,
        ctx_norm: f32,
        topk: usize,
    ) -> Option<usize> {
        let candidates = self.select_topk(logits, &self.relation, topk);
        if candidates.is_empty() {
            return None;
        }
        let mut best_idx = candidates[0];
        let mut best_score = f32::MIN;
        for idx in candidates {
            let logit_term = self.relation_weights.logit * logits[idx];
            let cos_term =
                self.relation_weights.cosine * self.cosine_with_context(idx, context, ctx_norm);
            let score = logit_term + cos_term;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    fn select_object(
        &self,
        logits: &ArrayView1<f32>,
        context_row: &ArrayView1<f32>,
        context_view: &ArrayView2<f32>,
        ctx_norm: f32,
        topk: usize,
    ) -> Option<usize> {
        let candidates = self.select_topk(logits, &self.object, topk);
        if candidates.is_empty() {
            return None;
        }
        let mut best_idx = candidates[0];
        let mut best_score = f32::MIN;
        for idx in candidates {
            let logit_term = self.object_weights.logit * logits[idx];
            let cos_term =
                self.object_weights.cosine * self.cosine_with_context(idx, context_row, ctx_norm);
            let geo_term =
                self.object_weights.geodesic * self.poincare_distance_single(idx, context_view);
            let score = logit_term + cos_term - geo_term;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    pub fn decode_batch(
        &self,
        logits: ArrayView2<f32>,
        relation_ctx: ArrayView2<f32>,
        object_ctx: ArrayView2<f32>,
        topk_relation: usize,
        topk_object: usize,
    ) -> Vec<usize> {
        assert_eq!(logits.nrows(), relation_ctx.nrows());
        assert_eq!(relation_ctx.nrows(), object_ctx.nrows());
        assert_eq!(relation_ctx.ncols(), self.embeddings.ncols());
        let batch = logits.nrows();
        let mut outputs = Vec::with_capacity(batch);
        for b in 0..batch {
            let log_row = logits.row(b);
            let rel_row = relation_ctx.row(b);
            let obj_row = object_ctx.row(b);
            let rel_norm = rel_row.dot(&rel_row).sqrt().max(EPS);
            let obj_norm = obj_row.dot(&obj_row).sqrt().max(EPS);
            let obj_view = object_ctx.slice(s![b..b + 1, ..]);
            let skel_choice = self.masked_argmax(&log_row, &self.skeleton);
            let rel_choice = self.select_relation(&log_row, &rel_row, rel_norm, topk_relation);
            let obj_choice =
                self.select_object(&log_row, &obj_row, &obj_view, obj_norm, topk_object);
            let token = obj_choice
                .or(rel_choice)
                .or(skel_choice)
                .unwrap_or_else(|| {
                    log_row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                });
            outputs.push(token);
        }
        outputs
    }
}
