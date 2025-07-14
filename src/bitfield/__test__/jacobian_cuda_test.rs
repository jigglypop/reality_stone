//! CUDA 비트 야코비안 테스트

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::bitfield::jacobian::{BitJacobian, HyperbolicBitJacobian};
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_cuda_cpu_일치성() {
        let mut jacobian_cpu = BitJacobian::new();
        let mut jacobian_gpu = BitJacobian::new();
        jacobian_gpu.set_use_cuda(true);

        // 테스트 데이터
        let codes = Array1::from(vec![
            0x000000FF, // sin, amp=1.0
            0x004000FF, // cos, amp=1.0
            0x008000FF, // -sin, amp=1.0
            0x00C000FF, // -cos, amp=1.0
        ]);

        let x = Array1::from(vec![0.0, 0.5, 1.0, 1.5]);

        // CPU 계산
        let cpu_result = jacobian_cpu.compute_jacobian(&codes, &x.view(), 0);

        // GPU 계산
        let gpu_result = jacobian_gpu.compute_jacobian(&codes, &x.view(), 0);

        // 결과 비교
        for i in 0..cpu_result.len() {
            assert_relative_eq!(cpu_result[i], gpu_result[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_cuda_미분_순환성() {
        let mut jacobian = BitJacobian::new();
        jacobian.set_use_cuda(true);

        let code = 0x000000FF; // sin, amp=1.0
        let codes = Array1::from(vec![code; 4]);
        let x = Array1::from(vec![std::f32::consts::PI / 4.0; 4]);

        // 각 미분 차수 계산
        let d0 = jacobian.compute_jacobian(&codes, &x.view(), 0)[0];
        let d1 = jacobian.compute_jacobian(&codes, &x.view(), 1)[0];
        let d2 = jacobian.compute_jacobian(&codes, &x.view(), 2)[0];
        let d3 = jacobian.compute_jacobian(&codes, &x.view(), 3)[0];
        let d4 = jacobian.compute_jacobian(&codes, &x.view(), 4)[0];

        // 순환성 확인
        assert_relative_eq!(d0, d4, epsilon = 1e-5);

        // 값 확인
        assert_relative_eq!(d0, 0.707, epsilon = 0.01); // sin(π/4)
        assert_relative_eq!(d1, 0.707, epsilon = 0.01); // cos(π/4)
        assert_relative_eq!(d2, -0.707, epsilon = 0.01); // -sin(π/4)
        assert_relative_eq!(d3, -0.707, epsilon = 0.01); // -cos(π/4)
    }

    #[test]
    fn test_cuda_대규모_배치() {
        let mut jacobian = BitJacobian::new();
        jacobian.set_use_cuda(true);

        // 대규모 데이터 (1024개)
        let n = 1024;
        let codes = Array1::from(vec![0x000000FF; n]);
        let x = Array1::linspace(0.0, 2.0 * std::f32::consts::PI, n);

        // GPU 계산
        let start = std::time::Instant::now();
        let gpu_result = jacobian.compute_jacobian(&codes, &x.view(), 0);
        let gpu_time = start.elapsed();

        // CPU 계산 (비교용)
        jacobian.set_use_cuda(false);
        let start = std::time::Instant::now();
        let cpu_result = jacobian.compute_jacobian(&codes, &x.view(), 0);
        let cpu_time = start.elapsed();

        println!("GPU 시간: {:?}", gpu_time);
        println!("CPU 시간: {:?}", cpu_time);
        println!(
            "속도 향상: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );

        // 결과 검증
        assert_eq!(gpu_result.len(), n);

        // 샘플링하여 일치성 확인
        for i in (0..n).step_by(100) {
            assert_relative_eq!(cpu_result[i], gpu_result[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_cuda_위상_변조() {
        let mut jacobian = BitJacobian::new();
        jacobian.set_use_cuda(true);

        // 다양한 위상값
        let phases = vec![0, 64, 128, 192, 255]; // 0, π/2, π, 3π/2, 2π
        let mut codes = Vec::new();

        for phase in phases {
            let code = (phase as u32) << 24 | 0x000000FF;
            codes.push(code);
        }

        let codes_array = Array1::from(codes);
        let x = Array1::zeros(5);

        let results = jacobian.compute_jacobian(&codes_array, &x.view(), 0);

        // sin(0 + phase) 값 확인
        assert_relative_eq!(results[0], 0.0, epsilon = 0.01); // sin(0)
        assert_relative_eq!(results[1], 1.0, epsilon = 0.01); // sin(π/2)
        assert_relative_eq!(results[2], 0.0, epsilon = 0.01); // sin(π)
        assert_relative_eq!(results[3], -1.0, epsilon = 0.01); // sin(3π/2)
        assert_relative_eq!(results[4], 0.0, epsilon = 0.01); // sin(2π)
    }

    #[test]
    fn test_cuda_진폭_정밀도() {
        let mut jacobian = BitJacobian::new();
        jacobian.set_use_cuda(true);

        // amp_fine 테스트
        let codes = Array1::from(vec![
            0x00000000, // amp=0, amp_fine=0
            0x00400080, // amp=128, amp_fine=1
            0x008000FF, // amp=255, amp_fine=2
            0x00C000FF, // amp=255, amp_fine=3
        ]);

        let x = Array1::zeros(4);
        let results = jacobian.compute_jacobian(&codes, &x.view(), 1); // cos(0) = 1

        // 예상 진폭: 0, 0.5+0.0625, 1.0+0.125, 1.0+0.1875
        assert_relative_eq!(results[0], 0.0, epsilon = 0.01);
        assert_relative_eq!(results[1], 0.5625, epsilon = 0.01);
        assert_relative_eq!(results[2], 1.125, epsilon = 0.01);
        assert_relative_eq!(results[3], 1.1875, epsilon = 0.01);
    }
}

// CUDA가 없을 때 스킵되는 테스트 표시
#[cfg(not(feature = "cuda"))]
#[test]
fn test_cuda_기능_비활성화() {
    println!("CUDA 테스트가 비활성화되었습니다. --features cuda로 활성화하세요.");
}
