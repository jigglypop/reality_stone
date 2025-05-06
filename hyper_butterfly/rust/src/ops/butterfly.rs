//! 버터플라이 변환 구현

use ndarray::{Array1, Array2, Axis};
use std::f32::consts::PI;

/// 버터플라이 변환 적용
pub fn butterfly_transform(
    input: &Array2<f32>,
    params: &Array1<f32>,
    num_layers: usize,
) -> Array2<f32> {
    let batch_size = input.shape()[0];
    let dim = input.shape()[1];
    let log2_dim = (dim as f32).log2() as usize;

    let mut output = input.clone();
    let mut param_offset = 0;

    for l in 0..num_layers {
        let layer_idx = l % log2_dim;
        let block_size = 1 << layer_idx;
        let num_blocks = dim / (2 * block_size);

        // 레이어별 파라미터 추출
        let layer_params = params.slice(s![param_offset..param_offset + num_blocks * 2]);
        param_offset += num_blocks * 2;

        // 현재 출력을 [batch, num_blocks, 2, block_size] 형태로 재구성
        let mut output_reshaped = output
            .clone()
            .into_shape((batch_size, num_blocks, 2, block_size))
            .unwrap();

        // 각 블록에 대해 변환 적용
        for b in 0..num_blocks {
            let a = layer_params[b * 2]; // 회전 파라미터 a
            let b_param = layer_params[b * 2 + 1]; // 회전 파라미터 b

            let x1 = output_reshaped.slice(s![.., b, 0, ..]).to_owned();
            let x2 = output_reshaped.slice(s![.., b, 1, ..]).to_owned();

            // 회전 변환 적용
            // y1 = a*x1 + b*x2
            // y2 = -b*x1 + a*x2
            let y1 = &x1 * a + &x2 * b_param;
            let y2 = &x1 * (-b_param) + &x2 * a;

            // 결과 저장
            for i in 0..batch_size {
                for j in 0..block_size {
                    output_reshaped[[i, b, 0, j]] = y1[[i, j]];
                    output_reshaped[[i, b, 1, j]] = y2[[i, j]];
                }
            }
        }

        // 출력 텐서 재구성
        output = output_reshaped.into_shape((batch_size, dim)).unwrap();
    }

    output
}

/// 무작위 버터플라이 파라미터 생성
pub fn random_butterfly_params(dim: usize, num_layers: usize) -> Array1<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let log2_dim = (dim as f32).log2() as usize;
    let mut total_params = 0;

    for l in 0..num_layers {
        let layer_idx = l % log2_dim;
        let block_size = 1 << layer_idx;
        let num_blocks = dim / (2 * block_size);
        total_params += num_blocks * 2; // 각 블록당 2개 파라미터
    }

    // 무작위 초기화 (작은 값으로)
    let mut params = Array1::zeros(total_params);
    for i in 0..total_params / 2 {
        // a 파라미터는 1에 가깝게 초기화
        params[2 * i] = 1.0 - rng.gen::<f32>() * 0.1;
        // b 파라미터는 0에 가깝게 초기화
        params[2 * i + 1] = (rng.gen::<f32>() - 0.5) * 0.1;
    }

    params
}
