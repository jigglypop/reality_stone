use crate::bitfield::{decoder, BitfieldLayout, BitfieldLinear};
use ndarray::{Array1, Array2};

pub mod jacobian_cuda_test;
pub mod jacobian_test;
pub mod riemannian_test;

#[test]
fn test_22bit_layout_encoding_decoding() {
    let cat = 2;
    let sub = 3;
    let idx = 129;
    let d = 3;
    let amp = 240;

    let code = decoder::encode_22bit(cat, sub, idx, d, amp);

    // 22비트이므로 상위 10비트는 0이어야 함
    assert_eq!(code >> 22, 0, "22비트 코드가 22비트를 초과함");

    let (d_cat, d_sub, d_idx, d_d, d_amp) = decoder::decode_22bit(code);

    assert_eq!(cat, d_cat, "Category 불일치");
    assert_eq!(sub, d_sub, "Sub-category 불일치");
    assert_eq!(idx, d_idx, "Index 불일치");
    assert_eq!(d, d_d, "D-param 불일치");
    assert_eq!(amp, d_amp, "Amplitude 불일치");
}

#[test]
fn test_bitfield_linear_22bit_creation() {
    let m = 10;
    let n = 20;
    let b = 32;
    let r_max = 2.0;
    let layout = BitfieldLayout::Extreme22Bit;

    let bitfield = BitfieldLinear::new(m, n, b, r_max, layout);

    assert_eq!(bitfield.get_m(), m);
    assert_eq!(bitfield.get_n(), n);
    assert_eq!(bitfield.layout, layout);
}

#[test]
fn test_bitfield_linear_forward_22bit() {
    let m = 4;
    let n = 8;
    let b = 16;
    let r_max = 1.0;
    let layout = BitfieldLayout::Extreme22Bit;

    let mut bitfield = BitfieldLinear::new(m, n, b, r_max, layout);

    // 수동으로 코드 설정 (예시)
    // cat=1, sub=1, idx=1, d=1, amp=128
    let code1 = decoder::encode_22bit(1, 1, 1, 1, 128);
    // cat=2, sub=0, idx=5, d=3, amp=200
    let code2 = decoder::encode_22bit(2, 0, 5, 3, 200);
    bitfield.codes = Array1::from(vec![code1, code2, 0, 0]);

    let x = Array2::<f32>::ones((2, n));
    let output = bitfield.forward(&x);

    assert_eq!(output.shape(), &[2, m]);
    // 특정 값 단언은 복잡하므로, 형태와 NaN/inf 여부만 확인
    assert!(!output.iter().any(|&v| v.is_nan() || v.is_infinite()));
}
