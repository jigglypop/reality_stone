/// 64-bit Packed Poincaré 시드 표현
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64(pub u64);

/// 기저 함수 타입
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,
    SinSinh = 1,
    CosCosh = 2,
    CosSinh = 3,
    BesselJ = 4,
    BesselI = 5,
    BesselK = 6,
    BesselY = 7,
    TanhSign = 8,
    SechTri = 9,
    ExpSin = 10,
    Morlet = 11,
}

/// 디코딩된 파라미터
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedParams {
    pub r: f32,           // 20 bits
    pub theta: f32,       // 24 bits
    pub basis_id: u8,     // 4 bits
    pub d_theta: u8,      // 2 bits - θ 미분 차수 (0-3)
    pub d_r: bool,        // 1 bit - r 미분 차수 (0 or 1)
    pub rot_code: u8,     // 4 bits - 회전 코드
    pub log2_c: i8,       // 3 bits - 곡률 (부호 있음)
    pub reserved: u8,     // 6 bits
}

/// 행렬 압축 및 복원
pub struct PoincareMatrix {
    pub seed: Packed64,
    pub rows: usize,
    pub cols: usize,
} 