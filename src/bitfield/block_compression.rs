//! 블록별 압축 시스템
//!
//! 큰 가중치 행렬을 작은 블록으로 나누어 압축하여 메모리 효율성 향상

use crate::bitfield::{
    adaptive::AdaptiveFunctionSelector, pipeline::PipelineConfig, BitfieldLayout,
};
use ndarray::{s, Array1, Array2, ArrayView2};

/// 블록 압축 설정
#[derive(Debug, Clone)]
pub struct BlockCompressionConfig {
    /// 블록 크기 (행 수)
    pub block_size: usize,

    /// 블록당 기저 벡터 수
    pub basis_per_block: usize,

    /// 블록 간 오버랩
    pub overlap: usize,

    /// 적응적 블록 크기 사용
    pub adaptive_block_size: bool,

    /// 파이프라인 설정
    pub pipeline_config: PipelineConfig,
}

impl Default for BlockCompressionConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            basis_per_block: 32,
            overlap: 0,
            adaptive_block_size: false,
            pipeline_config: PipelineConfig::default(),
        }
    }
}

/// 압축된 블록
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// 블록 시작 인덱스
    pub start_idx: usize,

    /// 블록 크기
    pub size: usize,

    /// 압축된 코드
    pub codes: Vec<u32>,

    /// 블록별 기저 테이블
    pub basis_table: Array2<f32>,

    /// 블록 통계
    pub stats: BlockStats,
}

/// 블록 통계
#[derive(Debug, Clone)]
pub struct BlockStats {
    /// 평균 재구성 에러
    pub avg_error: f32,

    /// 최대 재구성 에러
    pub max_error: f32,

    /// 블록 내 희소성
    pub sparsity: f32,

    /// 압축률
    pub compression_ratio: f32,
}

/// 블록 압축 시스템
pub struct BlockCompressor {
    config: BlockCompressionConfig,
    adaptive_selector: Option<AdaptiveFunctionSelector>,
}

impl BlockCompressor {
    /// 새로운 블록 압축기 생성
    pub fn new(config: BlockCompressionConfig) -> Self {
        let adaptive_selector = if config.pipeline_config.use_adaptive_selection {
            Some(AdaptiveFunctionSelector::new(
                config.pipeline_config.error_threshold,
            ))
        } else {
            None
        };

        Self {
            config,
            adaptive_selector,
        }
    }

    /// 가중치 행렬을 블록으로 압축
    pub fn compress_blocks(&mut self, weights: &Array2<f32>) -> Vec<CompressedBlock> {
        let (m, n) = weights.dim();
        let mut blocks = Vec::new();

        // 블록 크기 결정
        let block_sizes = if self.config.adaptive_block_size {
            self.compute_adaptive_block_sizes(weights)
        } else {
            vec![self.config.block_size; (m + self.config.block_size - 1) / self.config.block_size]
        };

        let mut current_idx = 0;

        for block_size in block_sizes {
            if current_idx >= m {
                break;
            }

            let end_idx = (current_idx + block_size).min(m);
            let block_view = weights.slice(s![current_idx..end_idx, ..]);

            // 블록 압축
            let compressed_block =
                self.compress_single_block(&block_view, current_idx, end_idx - current_idx);

            blocks.push(compressed_block);

            // 오버랩 처리
            current_idx = end_idx - self.config.overlap;
        }

        blocks
    }

    /// 단일 블록 압축
    fn compress_single_block(
        &mut self,
        block: &ArrayView2<f32>,
        start_idx: usize,
        size: usize,
    ) -> CompressedBlock {
        let (m, n) = block.dim();

        // 블록별 기저 생성
        let basis_table = self.generate_block_basis(block);

        // 가중치 압축
        let mut codes = Vec::new();
        let mut total_error = 0.0;
        let mut max_error = 0.0;
        let mut nonzero_count = 0;

        for row in block.rows() {
            // 희소성 계산
            nonzero_count += row.iter().filter(|&&x| x.abs() > 1e-6).count();

            // 압축
            let (code, error) = self.compress_row(&row, &basis_table);
            codes.push(code);

            total_error += error;
            max_error = max_error.max(error);
        }

        let avg_error = total_error / m as f32;
        let sparsity = 1.0 - (nonzero_count as f32) / (m * n) as f32;
        let compression_ratio =
            (m * n * 4) as f32 / (codes.len() * 4 + basis_table.len() * 4) as f32;

        CompressedBlock {
            start_idx,
            size,
            codes,
            basis_table,
            stats: BlockStats {
                avg_error,
                max_error,
                sparsity,
                compression_ratio,
            },
        }
    }

    /// 블록별 기저 생성
    fn generate_block_basis(&self, block: &ArrayView2<f32>) -> Array2<f32> {
        let (m, n) = block.dim();
        let b = self.config.basis_per_block.min(m).min(n);

        // SVD 기반 기저 생성
        if let Ok(svd) = block.svd(true, true) {
            if let Some(vt) = svd.vt {
                return vt.slice(s![..b, ..]).to_owned();
            }
        }

        // 폴백: PCA 기반 기저
        self.pca_basis(block, b)
    }

    /// PCA 기반 기저 생성
    fn pca_basis(&self, block: &ArrayView2<f32>, num_components: usize) -> Array2<f32> {
        let (m, n) = block.dim();

        // 평균 중심화
        let mean = block.mean_axis(ndarray::Axis(0)).unwrap();
        let centered = block - &mean.broadcast((m, n)).unwrap();

        // 공분산 행렬
        let cov = centered.t().dot(&centered) / (m - 1) as f32;

        // 고유값 분해
        if let Ok(eigen) = cov.eig() {
            let (eigenvalues, eigenvectors) = eigen;

            // 가장 큰 고유값에 해당하는 고유벡터 선택
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| eigenvalues[j].re.partial_cmp(&eigenvalues[i].re).unwrap());

            let mut basis = Array2::zeros((num_components, n));
            for (i, &idx) in indices.iter().take(num_components).enumerate() {
                for j in 0..n {
                    basis[[i, j]] = eigenvectors[[j, idx]].re;
                }
            }

            return basis;
        }

        // 폴백: 랜덤 기저
        Array2::from_shape_fn((num_components, n), |_| rand::random::<f32>() - 0.5)
    }

    /// 행 압축
    fn compress_row(&mut self, row: &ArrayView1<f32>, basis_table: &Array2<f32>) -> (u32, f32) {
        if let Some(ref mut selector) = self.adaptive_selector {
            // 적응적 선택
            let selection = selector.select_function(row, basis_table);
            let code = self.encode_selection(&selection);
            (code, selection.error)
        } else {
            // 기본 압축
            let mut best_idx = 0;
            let mut best_scale = 0.0;
            let mut best_error = f32::INFINITY;

            for (idx, basis_row) in basis_table.rows().into_iter().enumerate() {
                let projection = row.dot(&basis_row);
                let scale = projection / basis_row.dot(&basis_row);
                let reconstructed = &basis_row * scale;
                let error = (row - &reconstructed).dot(&(row - &reconstructed)).sqrt();

                if error < best_error {
                    best_error = error;
                    best_idx = idx;
                    best_scale = scale;
                }
            }

            let amp =
                ((best_scale / self.config.pipeline_config.r_max).clamp(0.0, 1.0) * 255.0) as u8;

            let code = match self.config.pipeline_config.layout {
                BitfieldLayout::Standard32Bit => (best_idx as u32) | ((amp as u32) << 8),
                BitfieldLayout::Extreme22Bit => ((best_idx as u32) << 10) | (amp as u32),
            };

            (code, best_error)
        }
    }

    /// 적응적 블록 크기 계산
    fn compute_adaptive_block_sizes(&self, weights: &Array2<f32>) -> Vec<usize> {
        let (m, _n) = weights.dim();
        let mut block_sizes = Vec::new();
        let mut current_idx = 0;

        while current_idx < m {
            // 현재 위치에서의 복잡도 추정
            let complexity = self.estimate_complexity(weights, current_idx);

            // 복잡도에 따른 블록 크기 결정
            let block_size = if complexity > 0.8 {
                self.config.block_size / 2 // 복잡한 영역: 작은 블록
            } else if complexity < 0.2 {
                self.config.block_size * 2 // 단순한 영역: 큰 블록
            } else {
                self.config.block_size // 중간 영역: 기본 크기
            };

            let block_size = block_size.min(m - current_idx);
            block_sizes.push(block_size);
            current_idx += block_size;
        }

        block_sizes
    }

    /// 복잡도 추정
    fn estimate_complexity(&self, weights: &Array2<f32>, start_idx: usize) -> f32 {
        let (m, n) = weights.dim();
        let end_idx = (start_idx + 32).min(m);

        if start_idx >= m {
            return 0.0;
        }

        let block = weights.slice(s![start_idx..end_idx, ..]);

        // 분산 기반 복잡도
        let variance = block.var(0.0);

        // 희소성 기반 복잡도
        let nonzero_ratio =
            block.iter().filter(|&&x| x.abs() > 1e-6).count() as f32 / block.len() as f32;

        // 복잡도 점수 (0~1)
        (variance * nonzero_ratio).tanh()
    }

    /// 선택 결과 인코딩
    fn encode_selection(&self, selection: &crate::bitfield::adaptive::AdaptiveSelection) -> u32 {
        match self.config.pipeline_config.layout {
            BitfieldLayout::Standard32Bit => {
                ((selection.cat as u32) << 20)
                    | ((selection.sub as u32) << 18)
                    | ((selection.idx as u32) << 10)
                    | ((selection.d as u32) << 8)
                    | (selection.amp as u32)
            }
            BitfieldLayout::Extreme22Bit => {
                ((selection.cat as u32) << 20)
                    | ((selection.sub as u32) << 18)
                    | ((selection.idx as u32) << 10)
                    | ((selection.d as u32) << 8)
                    | (selection.amp as u32)
            }
        }
    }
}

/// 블록 압축 결과 병합
pub fn merge_blocks(blocks: &[CompressedBlock], n: usize) -> (Vec<u32>, Array2<f32>) {
    let total_rows: usize = blocks.iter().map(|b| b.size).sum();
    let mut all_codes = Vec::with_capacity(total_rows);
    let mut merged_basis = Vec::new();

    // 코드와 기저 병합
    for block in blocks {
        all_codes.extend(&block.codes);
        for row in block.basis_table.rows() {
            merged_basis.push(row.to_vec());
        }
    }

    // 기저 테이블 생성
    let basis_rows = merged_basis.len();
    let mut basis_array = Array2::zeros((basis_rows, n));
    for (i, row) in merged_basis.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j < n {
                basis_array[[i, j]] = val;
            }
        }
    }

    (all_codes, basis_array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_블록_압축_기본() {
        let weights = Array2::from_shape_fn((256, 128), |(i, j)| {
            (i as f32 * 0.01 + j as f32 * 0.001).sin()
        });

        let config = BlockCompressionConfig::default();
        let mut compressor = BlockCompressor::new(config);

        let blocks = compressor.compress_blocks(&weights);

        assert!(!blocks.is_empty());
        println!("압축된 블록 수: {}", blocks.len());

        for (i, block) in blocks.iter().enumerate() {
            println!(
                "블록 {}: 크기={}, 압축률={:.2}x, 평균 에러={:.6}",
                i, block.size, block.stats.compression_ratio, block.stats.avg_error
            );
        }
    }

    #[test]
    fn test_적응적_블록_크기() {
        let mut weights = Array2::zeros((300, 64));

        // 복잡한 영역
        for i in 0..100 {
            for j in 0..64 {
                weights[[i, j]] = ((i * j) as f32 * 0.1).sin();
            }
        }

        // 단순한 영역
        for i in 200..300 {
            for j in 0..32 {
                weights[[i, j]] = 0.1;
            }
        }

        let mut config = BlockCompressionConfig::default();
        config.adaptive_block_size = true;

        let mut compressor = BlockCompressor::new(config);
        let blocks = compressor.compress_blocks(&weights);

        // 블록 크기 변화 확인
        for block in &blocks {
            println!(
                "블록 크기: {}, 희소성: {:.2}",
                block.size, block.stats.sparsity
            );
        }
    }

    #[test]
    fn test_블록_병합() {
        let weights = Array2::from_shape_fn((200, 100), |(i, j)| (i as f32 - j as f32) * 0.01);

        let config = BlockCompressionConfig {
            block_size: 50,
            ..Default::default()
        };

        let mut compressor = BlockCompressor::new(config);
        let blocks = compressor.compress_blocks(&weights);

        let (codes, basis) = merge_blocks(&blocks, 100);

        assert_eq!(codes.len(), 200);
        println!("병합된 기저 크기: {:?}", basis.dim());
    }
}
