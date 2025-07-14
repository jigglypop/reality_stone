//! 적응적 함수 선택 시스템
//!
//! 가중치의 특성에 따라 최적의 리만 기하학 함수를 자동으로 선택

use crate::bitfield::riemannian::get_riemannian_function;
use ndarray::{s, Array1, Array2, ArrayView1};

/// 적응적 함수 선택 결과
#[derive(Debug, Clone)]
pub struct AdaptiveSelection {
    pub cat: u8,
    pub sub: u8,
    pub idx: u8,
    pub d: u8,
    pub amp: u8,
    pub error: f32,
}

/// 적응적 함수 선택기
pub struct AdaptiveFunctionSelector {
    /// 함수 선택 통계
    selection_stats: [[u32; 4]; 4], // [cat][sub] 카운트

    /// 에러 임계값
    error_threshold: f32,

    /// 빠른 선택 모드
    fast_mode: bool,
}

impl AdaptiveFunctionSelector {
    /// 새로운 선택기 생성
    pub fn new(error_threshold: f32) -> Self {
        Self {
            selection_stats: [[0; 4]; 4],
            error_threshold,
            fast_mode: false,
        }
    }

    /// 빠른 선택 모드 설정
    pub fn set_fast_mode(&mut self, fast: bool) {
        self.fast_mode = fast;
    }

    /// 가중치 행에 대한 최적 함수 선택
    pub fn select_function(
        &mut self,
        weight_row: &ArrayView1<f32>,
        basis_table: &Array2<f32>,
    ) -> AdaptiveSelection {
        // 1단계: 가중치 특성 분석
        let characteristics = self.analyze_weights(weight_row);

        // 2단계: 최적 기저 벡터 찾기
        let (best_idx, best_projection) = self.find_best_basis(weight_row, basis_table);

        // 3단계: 함수 카테고리 선택
        let (cat, sub) = if self.fast_mode {
            self.fast_category_selection(&characteristics)
        } else {
            self.exhaustive_category_selection(
                weight_row,
                &basis_table.row(best_idx),
                best_projection,
            )
        };

        // 4단계: 세부 파라미터 최적화
        let (d, amp, error) = self.optimize_parameters(
            weight_row,
            &basis_table.row(best_idx),
            cat,
            sub,
            best_projection,
        );

        // 통계 업데이트
        self.selection_stats[cat as usize][sub as usize] += 1;

        AdaptiveSelection {
            cat,
            sub,
            idx: best_idx as u8,
            d,
            amp,
            error,
        }
    }

    /// 가중치 특성 분석
    fn analyze_weights(&self, weights: &ArrayView1<f32>) -> WeightCharacteristics {
        let norm = weights.dot(weights).sqrt();
        let mean = weights.mean().unwrap_or(0.0);
        let variance = weights.var(0.0);

        // 희소성 측정 (0이 아닌 요소 비율)
        let sparsity =
            weights.iter().filter(|&&x| x.abs() > 1e-6).count() as f32 / weights.len() as f32;

        // 주기성 검사 (간단한 FFT 대체)
        let periodicity = self.estimate_periodicity(weights);

        // 대칭성 검사
        let symmetry = self.check_symmetry(weights);

        WeightCharacteristics {
            norm,
            mean,
            variance,
            sparsity,
            periodicity,
            symmetry,
        }
    }

    /// 최적 기저 벡터 찾기
    fn find_best_basis(
        &self,
        weight_row: &ArrayView1<f32>,
        basis_table: &Array2<f32>,
    ) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_projection = 0.0f32;

        for (idx, basis_row) in basis_table.rows().into_iter().enumerate() {
            let projection = weight_row.dot(&basis_row).abs();
            if projection > best_projection {
                best_projection = projection;
                best_idx = idx;
            }
        }

        (best_idx, best_projection)
    }

    /// 빠른 카테고리 선택
    fn fast_category_selection(&self, chars: &WeightCharacteristics) -> (u8, u8) {
        if chars.norm < 0.5 {
            // 작은 노름 → Poincaré
            (0, 0)
        } else if chars.norm > 3.0 {
            // 큰 노름 → Lorentz
            (1, 0)
        } else if chars.sparsity < 0.3 {
            // 희소 → Klein
            (2, 0)
        } else if chars.periodicity > 0.5 {
            // 주기적 → 특수 함수
            (3, 2)
        } else {
            // 기본값
            (0, 1)
        }
    }

    /// 전체 카테고리 탐색
    fn exhaustive_category_selection(
        &self,
        weight_row: &ArrayView1<f32>,
        basis_row: &ArrayView1<f32>,
        projection: f32,
    ) -> (u8, u8) {
        let mut best_cat = 0;
        let mut best_sub = 0;
        let mut best_error = f32::INFINITY;

        // 모든 카테고리/서브카테고리 조합 테스트
        for cat in 0..4 {
            for sub in 0..4 {
                // 대표 함수로 빠른 테스트
                let test_r = projection;
                let scale = get_riemannian_function(cat, sub, 0, test_r);
                let reconstructed = basis_row * scale;
                let error = (weight_row - &reconstructed)
                    .dot(&(weight_row - &reconstructed))
                    .sqrt();

                if error < best_error {
                    best_error = error;
                    best_cat = cat;
                    best_sub = sub;
                }
            }
        }

        (best_cat, best_sub)
    }

    /// 세부 파라미터 최적화
    fn optimize_parameters(
        &self,
        weight_row: &ArrayView1<f32>,
        basis_row: &ArrayView1<f32>,
        cat: u8,
        sub: u8,
        initial_r: f32,
    ) -> (u8, u8, f32) {
        let mut best_d = 0;
        let mut best_amp = 0;
        let mut best_error = f32::INFINITY;

        // 모든 d 값 테스트
        for d in 0..4 {
            // 이진 탐색으로 최적 r 찾기
            let optimal_r = self.binary_search_r(weight_row, basis_row, cat, sub, d, initial_r);

            // 양자화
            let amp = ((optimal_r / 2.0).clamp(0.0, 1.0) * 255.0) as u8;
            let quantized_r = (amp as f32) * 2.0 / 255.0;

            // 에러 계산
            let scale = get_riemannian_function(cat, sub, d, quantized_r);
            let reconstructed = basis_row * scale;
            let error = (weight_row - &reconstructed)
                .dot(&(weight_row - &reconstructed))
                .sqrt();

            if error < best_error {
                best_error = error;
                best_d = d;
                best_amp = amp;
            }
        }

        (best_d, best_amp, best_error)
    }

    /// 이진 탐색으로 최적 r 찾기
    fn binary_search_r(
        &self,
        weight_row: &ArrayView1<f32>,
        basis_row: &ArrayView1<f32>,
        cat: u8,
        sub: u8,
        d: u8,
        initial: f32,
    ) -> f32 {
        let mut low = 0.0;
        let mut high = 2.0;
        let mut best_r = initial.clamp(0.0, 2.0);

        // 초기값 근처에서 시작
        if initial > 0.0 && initial < 2.0 {
            low = (initial - 0.5).max(0.0);
            high = (initial + 0.5).min(2.0);
        }

        for _ in 0..10 {
            let mid1 = low + (high - low) / 3.0;
            let mid2 = high - (high - low) / 3.0;

            let scale1 = get_riemannian_function(cat, sub, d, mid1);
            let scale2 = get_riemannian_function(cat, sub, d, mid2);

            let error1 =
                (weight_row - &(basis_row * scale1)).dot(&(weight_row - &(basis_row * scale1)));
            let error2 =
                (weight_row - &(basis_row * scale2)).dot(&(weight_row - &(basis_row * scale2)));

            if error1 < error2 {
                high = mid2;
                best_r = mid1;
            } else {
                low = mid1;
                best_r = mid2;
            }

            if high - low < 0.01 {
                break;
            }
        }

        best_r
    }

    /// 주기성 추정
    fn estimate_periodicity(&self, weights: &ArrayView1<f32>) -> f32 {
        if weights.len() < 4 {
            return 0.0;
        }

        // 간단한 자기상관 기반 주기성 검사
        let mut max_correlation = 0.0;
        let max_lag = weights.len() / 4;

        for lag in 1..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..weights.len() - lag {
                correlation += weights[i] * weights[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                max_correlation = max_correlation.max(correlation.abs());
            }
        }

        max_correlation
    }

    /// 대칭성 검사
    fn check_symmetry(&self, weights: &ArrayView1<f32>) -> f32 {
        let n = weights.len();
        if n < 2 {
            return 0.0;
        }

        let mut symmetry_score = 0.0;
        let half = n / 2;

        for i in 0..half {
            let diff = (weights[i] - weights[n - 1 - i]).abs();
            symmetry_score += diff;
        }

        1.0 - (symmetry_score / (half as f32))
    }

    /// 선택 통계 가져오기
    pub fn get_selection_stats(&self) -> &[[u32; 4]; 4] {
        &self.selection_stats
    }
}

/// 가중치 특성
#[derive(Debug, Clone)]
struct WeightCharacteristics {
    norm: f32,
    mean: f32,
    variance: f32,
    sparsity: f32,
    periodicity: f32,
    symmetry: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_적응적_선택_기본() {
        let mut selector = AdaptiveFunctionSelector::new(1e-3);

        // 테스트 가중치
        let weights = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let basis = Array2::eye(4);

        let selection = selector.select_function(&weights.view(), &basis);

        assert!(selection.error < 1.0);
        println!(
            "선택된 함수: cat={}, sub={}, d={}",
            selection.cat, selection.sub, selection.d
        );
    }

    #[test]
    fn test_빠른_모드() {
        let mut selector = AdaptiveFunctionSelector::new(1e-3);
        selector.set_fast_mode(true);

        let weights = Array1::linspace(0.0, 1.0, 100);
        let mut basis = Array2::zeros((256, 100));
        for i in 0..100 {
            basis[[i, i]] = 1.0;
        }

        let selection = selector.select_function(&weights.view(), &basis);

        println!("빠른 선택 결과: {:?}", selection);
    }

    #[test]
    fn test_주기성_감지() {
        let selector = AdaptiveFunctionSelector::new(1e-3);

        // 주기적 신호
        let mut weights = Array1::zeros(64);
        for i in 0..64 {
            weights[i] = (i as f32 * 2.0 * std::f32::consts::PI / 16.0).sin();
        }

        let periodicity = selector.estimate_periodicity(&weights.view());
        assert!(periodicity > 0.5);
        println!("주기성 점수: {}", periodicity);
    }
}
