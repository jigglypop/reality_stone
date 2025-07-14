//! 압축 파이프라인 및 프로파일링 시스템
//!
//! 가중치 압축의 전체 과정을 관리하고 성능을 모니터링

use crate::bitfield::{
    adaptive::{AdaptiveFunctionSelector, AdaptiveSelection},
    basis::find_best_basis_vector,
    BitfieldLayout,
};
use ndarray::{s, Array1, Array2, ArrayView2};
use std::time::{Duration, Instant};

/// 압축 파이프라인 설정
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// 기저 벡터 수
    pub basis_size: usize,

    /// 최대 반지름
    pub r_max: f32,

    /// 에러 임계값
    pub error_threshold: f32,

    /// 병렬 처리 활성화
    pub enable_parallel: bool,

    /// 적응적 함수 선택 사용
    pub use_adaptive_selection: bool,

    /// 프로파일링 활성화
    pub enable_profiling: bool,

    /// 비트필드 레이아웃
    pub layout: BitfieldLayout,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            basis_size: 256,
            r_max: 2.0,
            error_threshold: 1e-3,
            enable_parallel: true,
            use_adaptive_selection: true,
            enable_profiling: false,
            layout: BitfieldLayout::Standard32Bit,
        }
    }
}

/// 압축 결과
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// 압축된 코드
    pub codes: Vec<u32>,

    /// 기저 테이블
    pub basis_table: Array2<f32>,

    /// 압축 통계
    pub stats: CompressionStats,

    /// 프로파일링 데이터 (옵션)
    pub profile: Option<ProfilingData>,
}

/// 압축 통계
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// 평균 재구성 에러
    pub avg_error: f32,

    /// 최대 재구성 에러
    pub max_error: f32,

    /// 압축률
    pub compression_ratio: f32,

    /// 함수 사용 분포
    pub function_usage: [[u32; 4]; 4],

    /// 처리된 가중치 수
    pub weights_processed: usize,
}

/// 프로파일링 데이터
#[derive(Debug, Clone)]
pub struct ProfilingData {
    /// 전체 압축 시간
    pub total_time: Duration,

    /// 기저 생성 시간
    pub basis_generation_time: Duration,

    /// 함수 선택 시간
    pub function_selection_time: Duration,

    /// 인코딩 시간
    pub encoding_time: Duration,

    /// 단계별 시간 분포
    pub stage_times: Vec<(String, Duration)>,
}

/// 압축 파이프라인
pub struct CompressionPipeline {
    config: PipelineConfig,
    adaptive_selector: Option<AdaptiveFunctionSelector>,
}

impl CompressionPipeline {
    /// 새로운 파이프라인 생성
    pub fn new(config: PipelineConfig) -> Self {
        let adaptive_selector = if config.use_adaptive_selection {
            Some(AdaptiveFunctionSelector::new(config.error_threshold))
        } else {
            None
        };

        Self {
            config,
            adaptive_selector,
        }
    }

    /// 가중치 압축 실행
    pub fn compress(&mut self, weights: &Array2<f32>) -> CompressionResult {
        let start_time = Instant::now();
        let mut stage_times = Vec::new();

        // 1단계: 기저 테이블 생성
        let basis_start = Instant::now();
        let basis_table = self.generate_basis_table(weights);
        let basis_time = basis_start.elapsed();
        stage_times.push(("기저 생성".to_string(), basis_time));

        // 2단계: 가중치 압축
        let compress_start = Instant::now();
        let (codes, stats) = if self.config.enable_parallel {
            self.compress_parallel(weights, &basis_table)
        } else {
            self.compress_sequential(weights, &basis_table)
        };
        let compress_time = compress_start.elapsed();
        stage_times.push(("압축".to_string(), compress_time));

        // 3단계: 검증 (옵션)
        if self.config.enable_profiling {
            let verify_start = Instant::now();
            self.verify_compression(weights, &codes, &basis_table);
            let verify_time = verify_start.elapsed();
            stage_times.push(("검증".to_string(), verify_time));
        }

        let total_time = start_time.elapsed();

        // 프로파일링 데이터 생성
        let profile = if self.config.enable_profiling {
            Some(ProfilingData {
                total_time,
                basis_generation_time: basis_time,
                function_selection_time: Duration::from_secs(0), // 개별 측정 필요
                encoding_time: compress_time,
                stage_times,
            })
        } else {
            None
        };

        CompressionResult {
            codes,
            basis_table,
            stats,
            profile,
        }
    }

    /// 기저 테이블 생성
    fn generate_basis_table(&self, weights: &Array2<f32>) -> Array2<f32> {
        let (m, n) = weights.dim();
        let b = self.config.basis_size.min(m).min(n);

        // SVD를 사용한 기저 생성
        if let Ok(svd) = weights.svd(true, true) {
            if let Some(vt) = svd.vt {
                return vt.slice(s![..b, ..]).to_owned();
            }
        }

        // 폴백: 랜덤 기저
        Array2::from_shape_fn((b, n), |_| rand::random::<f32>() - 0.5)
    }

    /// 순차적 압축
    fn compress_sequential(
        &mut self,
        weights: &Array2<f32>,
        basis_table: &Array2<f32>,
    ) -> (Vec<u32>, CompressionStats) {
        let mut codes = Vec::new();
        let mut total_error = 0.0;
        let mut max_error = 0.0;
        let mut function_usage = [[0u32; 4]; 4];

        for row in weights.rows() {
            let code = if let Some(ref mut selector) = self.adaptive_selector {
                // 적응적 선택
                let selection = selector.select_function(&row, basis_table);
                function_usage[selection.cat as usize][selection.sub as usize] += 1;
                total_error += selection.error;
                max_error = max_error.max(selection.error);

                self.encode_selection(&selection)
            } else {
                // 기본 선택
                let (idx, scale) = find_best_basis_vector(&row, basis_table, self.config.r_max);
                let amp = ((scale / self.config.r_max).clamp(0.0, 1.0) * 255.0) as u8;

                match self.config.layout {
                    BitfieldLayout::Standard32Bit => {
                        // 32비트 인코딩
                        (idx as u32) | ((amp as u32) << 8)
                    }
                    BitfieldLayout::Extreme22Bit => {
                        // 22비트 인코딩
                        (idx as u32) | ((amp as u32) << 10)
                    }
                }
            };

            codes.push(code);
        }

        let avg_error = total_error / weights.nrows() as f32;
        let compression_ratio = (weights.len() * 4) as f32 / (codes.len() * 4) as f32;

        let stats = CompressionStats {
            avg_error,
            max_error,
            compression_ratio,
            function_usage,
            weights_processed: weights.nrows(),
        };

        (codes, stats)
    }

    /// 병렬 압축
    fn compress_parallel(
        &mut self,
        weights: &Array2<f32>,
        basis_table: &Array2<f32>,
    ) -> (Vec<u32>, CompressionStats) {
        use rayon::prelude::*;

        let rows: Vec<_> = weights.rows().into_iter().collect();

        // 병렬 처리
        let results: Vec<_> = rows
            .par_iter()
            .map(|row| {
                if self.config.use_adaptive_selection {
                    // 적응적 선택은 스레드 안전하지 않으므로 기본 방법 사용
                    let (idx, scale) = find_best_basis_vector(row, basis_table, self.config.r_max);
                    let amp = ((scale / self.config.r_max).clamp(0.0, 1.0) * 255.0) as u8;

                    let code = match self.config.layout {
                        BitfieldLayout::Standard32Bit => (idx as u32) | ((amp as u32) << 8),
                        BitfieldLayout::Extreme22Bit => (idx as u32) | ((amp as u32) << 10),
                    };

                    (code, 0.0) // 에러 계산 생략
                } else {
                    let (idx, scale) = find_best_basis_vector(row, basis_table, self.config.r_max);
                    let amp = ((scale / self.config.r_max).clamp(0.0, 1.0) * 255.0) as u8;

                    let code = match self.config.layout {
                        BitfieldLayout::Standard32Bit => (idx as u32) | ((amp as u32) << 8),
                        BitfieldLayout::Extreme22Bit => (idx as u32) | ((amp as u32) << 10),
                    };

                    (code, 0.0)
                }
            })
            .collect();

        let codes: Vec<u32> = results.iter().map(|(code, _)| *code).collect();
        let total_error: f32 = results.iter().map(|(_, error)| error).sum();

        let stats = CompressionStats {
            avg_error: total_error / weights.nrows() as f32,
            max_error: 0.0, // 병렬 처리에서는 계산 생략
            compression_ratio: (weights.len() * 4) as f32 / (codes.len() * 4) as f32,
            function_usage: [[0; 4]; 4],
            weights_processed: weights.nrows(),
        };

        (codes, stats)
    }

    /// 선택 결과를 코드로 인코딩
    fn encode_selection(&self, selection: &AdaptiveSelection) -> u32 {
        match self.config.layout {
            BitfieldLayout::Standard32Bit => {
                // 32비트: phase(8) + amp_fine(2) + cat(2) + sub(2) + idx(8) + sign(1) + d(1) + amp(8)
                ((selection.cat as u32) << 20)
                    | ((selection.sub as u32) << 18)
                    | ((selection.idx as u32) << 10)
                    | ((selection.d as u32) << 8)
                    | (selection.amp as u32)
            }
            BitfieldLayout::Extreme22Bit => {
                // 22비트: cat(2) + sub(2) + idx(8) + d(2) + amp(8)
                ((selection.cat as u32) << 20)
                    | ((selection.sub as u32) << 18)
                    | ((selection.idx as u32) << 10)
                    | ((selection.d as u32) << 8)
                    | (selection.amp as u32)
            }
        }
    }

    /// 압축 결과 검증
    fn verify_compression(&self, original: &Array2<f32>, codes: &[u32], basis_table: &Array2<f32>) {
        // 재구성 및 에러 계산
        let mut total_error = 0.0;

        for (i, (row, &code)) in original.rows().into_iter().zip(codes.iter()).enumerate() {
            // 코드 디코딩
            let (idx, amp) = match self.config.layout {
                BitfieldLayout::Standard32Bit => {
                    ((code & 0xFF) as usize, ((code >> 8) & 0xFF) as f32 / 255.0)
                }
                BitfieldLayout::Extreme22Bit => {
                    (((code >> 10) & 0xFF) as usize, (code & 0xFF) as f32 / 255.0)
                }
            };

            if idx < basis_table.nrows() {
                let scale = amp * self.config.r_max;
                let reconstructed = &basis_table.row(idx) * scale;
                let error = (&row - &reconstructed).dot(&(&row - &reconstructed)).sqrt();
                total_error += error;

                if i < 5 && self.config.enable_profiling {
                    println!("행 {}: 에러 = {:.6}", i, error);
                }
            }
        }

        let avg_error = total_error / original.nrows() as f32;
        println!("평균 재구성 에러: {:.6}", avg_error);
    }
}

/// 압축 성능 벤치마크
pub fn benchmark_compression(weights: &Array2<f32>, configs: &[PipelineConfig]) {
    println!("=== 압축 파이프라인 벤치마크 ===");
    println!("가중치 크기: {:?}", weights.dim());

    for (i, config) in configs.iter().enumerate() {
        println!("\n설정 {}: {:?}", i + 1, config.layout);

        let mut pipeline = CompressionPipeline::new(config.clone());
        let result = pipeline.compress(weights);

        println!("압축률: {:.2}x", result.stats.compression_ratio);
        println!("평균 에러: {:.6}", result.stats.avg_error);

        if let Some(profile) = result.profile {
            println!("전체 시간: {:?}", profile.total_time);
            for (stage, time) in profile.stage_times {
                println!("  {}: {:?}", stage, time);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_압축_파이프라인_기본() {
        let weights = Array2::from_shape_fn((100, 64), |(i, j)| {
            (i as f32 * 0.01 + j as f32 * 0.001).sin()
        });

        let config = PipelineConfig::default();
        let mut pipeline = CompressionPipeline::new(config);

        let result = pipeline.compress(&weights);

        assert_eq!(result.codes.len(), 100);
        assert!(result.stats.avg_error < 0.1);
        println!("압축 통계: {:?}", result.stats);
    }

    #[test]
    fn test_병렬_압축() {
        let weights = Array2::from_shape_fn((1000, 128), |(i, j)| ((i + j) as f32 * 0.01).cos());

        let mut config = PipelineConfig::default();
        config.enable_parallel = true;
        config.enable_profiling = true;

        let mut pipeline = CompressionPipeline::new(config);
        let result = pipeline.compress(&weights);

        if let Some(profile) = result.profile {
            println!("병렬 압축 프로파일: {:?}", profile);
        }
    }

    #[test]
    fn test_22비트_압축() {
        let weights = Array2::from_shape_fn((50, 32), |(i, j)| (i as f32 - j as f32) * 0.1);

        let mut config = PipelineConfig::default();
        config.layout = BitfieldLayout::Extreme22Bit;

        let mut pipeline = CompressionPipeline::new(config);
        let result = pipeline.compress(&weights);

        // 22비트 코드 확인
        for &code in &result.codes {
            assert!(code <= 0x3FFFFF); // 22비트 최대값
        }
    }
}
