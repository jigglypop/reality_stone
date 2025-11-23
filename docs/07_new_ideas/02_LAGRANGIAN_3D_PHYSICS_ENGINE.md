# L-3DGS: 라그랑지안 3D 가우시안 스플래팅 물리 엔진

## 1. 핵심 아이디어

3D Gaussian Splatting의 공분산 행렬 $\Sigma$를 리만 메트릭 텐서의 역행렬 $g^{-1}$로 해석하여, 렌더링과 물리 시뮬레이션을 단일 데이터 구조로 통합합니다. 메시(Mesh) 없이 기하학적 변형, 충돌, 유체를 실시간으로 시뮬레이션합니다.

## 2. 수학적 기반

### 2.1 가우시안 스플랫과 리만 메트릭의 이중성

**3D Gaussian Splatting**:

각 가우시안은 $(μ, Σ)$로 정의:

$$
G(x) = \exp\left(-\frac{1}{2}(x-μ)^T Σ^{-1} (x-μ)\right)
$$

**리만 메트릭**:

점 $μ$에서의 거리 제곱:

$$
d^2(x, μ) = (x-μ)^T g(μ) (x-μ)
$$

**관계식**:

$$
g(μ) = Σ^{-1}
$$

즉, 가우시안의 공분산 = 공간의 메트릭 역행렬입니다.

### 2.2 라그랑지안 역학

각 가우시안 $i$의 중심 $μ_i$와 공분산 $Σ_i$가 시간에 따라 진화:

**운동 에너지**:

$$
T = \sum_i \frac{1}{2} m_i \|\dot{μ}_i\|^2 + \frac{1}{2} \text{tr}(\dot{Σ}_i^T M_i \dot{Σ}_i)
$$

여기서 $M_i$는 공분산의 "관성 텐서".

**잠재 에너지**:

$$
V = \sum_i V_{\text{grav}}(μ_i) + \sum_{i<j} V_{\text{collision}}(μ_i, μ_j, Σ_i, Σ_j) + \sum_i V_{\text{strain}}(Σ_i)
$$

- $V_{\text{grav}}$: 중력 포텐셜
- $V_{\text{collision}}$: 충돌 포텐셜 (가우시안 겹침 기반)
- $V_{\text{strain}}$: 변형 에너지 (고체 물체의 탄성)

**오일러-라그랑주 방정식**:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{μ}_i} = \frac{\partial L}{\partial μ_i}, \quad
\frac{d}{dt}\frac{\partial L}{\partial \dot{Σ}_i} = \frac{\partial L}{\partial Σ_i}
$$

### 2.3 메트릭 흐름 = 변형

물체가 힘을 받으면 공분산이 변화 = 메트릭이 변화:

$$
\frac{d Σ_i}{dt} = F_{\text{ext}}(μ_i) + \text{Damping}(Σ_i)
$$

**예시**:
- 압축력: $Σ$ 축소 (공간 수축)
- 인장력: $Σ$ 확대 (공간 팽창)
- 전단력: $Σ$ 비대각 성분 변화 (공간 비틀림)

## 3. 아키텍처 설계

### 3.1 데이터 구조

**GaussianSplat**:

```rust
struct GaussianSplat {
    position: Vec3,      // μ
    covariance: Mat3,    // Σ (SPD 행렬)
    color: Vec3,         // RGB
    opacity: f32,        // α
    mass: f32,           // 물리 질량
    velocity: Vec3,      // 속도
    angular_vel: Vec3,   // 회전 속도
}
```

**Scene**:

```rust
struct Scene {
    gaussians: Vec<GaussianSplat>,  // N개 가우시안
    spatial_hash: HashMap<GridCell, Vec<usize>>,  // 충돌 감지용
    forces: Vec<Box<dyn Force>>,    // 중력, 스프링 등
}
```

### 3.2 렌더링 파이프라인 (wgpu)

**Rasterization Compute Shader**:

```wgsl
@compute @workgroup_size(16, 16)
fn render_gaussians(
    @builtin(global_invocation_id) pixel: vec3<u32>
) {
    var color = vec3<f32>(0.0);
    var depth = 1e10;
    
    for (var i = 0u; i < num_gaussians; i++) {
        let g = gaussians[i];
        let dx = pixel_to_world(pixel) - g.position;
        
        // Mahalanobis 거리 (리만 거리)
        let Sigma_inv = inverse(g.covariance);
        let dist2 = dot(dx, Sigma_inv * dx);
        
        if dist2 < 9.0 {  // 3-sigma 컷오프
            let weight = exp(-0.5 * dist2);
            color += g.color * weight * g.opacity;
            depth = min(depth, length(dx));
        }
    }
    
    output[pixel.xy] = vec4<f32>(color, 1.0);
}
```

**렌더링 복잡도**: $O(N \cdot P)$ (N = 가우시안 수, P = 픽셀 수)

Spatial hashing으로 $O(k \cdot P)$로 감소 (k = 픽셀당 평균 가우시안 수, 보통 10-50).

### 3.3 물리 시뮬레이션 (Rust)

**Symplectic Integrator** (에너지 보존):

```rust
fn update_physics(scene: &mut Scene, dt: f32) {
    // 1. 힘 계산
    let forces = compute_forces(scene);
    
    // 2. Velocity Verlet
    for (i, g) in scene.gaussians.iter_mut().enumerate() {
        // 위치 업데이트 (반쪽)
        g.position += g.velocity * (dt * 0.5);
        
        // 속도 업데이트
        let accel = forces[i] / g.mass;
        g.velocity += accel * dt;
        
        // 위치 업데이트 (나머지 반쪽)
        g.position += g.velocity * (dt * 0.5);
        
        // 공분산 업데이트 (메트릭 변형)
        let strain = compute_strain(g, &forces[i]);
        g.covariance = update_covariance(g.covariance, strain, dt);
        
        // SPD 제약 보장
        g.covariance = project_to_spd(g.covariance);
    }
    
    // 3. 충돌 해결
    resolve_collisions(scene);
}
```

**SPD 투영**:

공분산이 SPD(Symmetric Positive Definite)를 벗어나면 고유값 분해 후 음수 고유값 제거:

```rust
fn project_to_spd(Sigma: Mat3) -> Mat3 {
    let (U, lambda, _) = eigen_decompose(Sigma);
    let lambda_pos = lambda.map(|x| x.max(1e-6));
    U * diag(lambda_pos) * U.transpose()
}
```

### 3.4 충돌 감지

**가우시안 겹침 기반**:

두 가우시안 $i, j$가 충돌하는 조건:

$$
\text{Overlap}(i, j) = \int G_i(x) G_j(x) dx > \text{threshold}
$$

닫힌 형태 해:

$$
\text{Overlap}(i, j) = \exp\left(-\frac{1}{2}(μ_i - μ_j)^T (Σ_i + Σ_j)^{-1} (μ_i - μ_j)\right)
$$

**충돌 힘**:

$$
F_{\text{collision}} = k_{\text{col}} \cdot \text{Overlap}(i,j) \cdot \frac{μ_i - μ_j}{\|μ_i - μ_j\|}
$$

### 3.5 변형 에너지 (탄성체)

**Neo-Hookean 모델**:

$$
V_{\text{strain}}(Σ) = \frac{\mu}{2}\left(\text{tr}(Σ Σ_0^{-1}) - 3\right) + \frac{\lambda}{2}\left(\log\det(Σ Σ_0^{-1})\right)^2
$$

여기서 $Σ_0$는 rest 공분산, $\mu, \lambda$는 라메 상수.

**그라디언트**:

$$
\frac{\partial V_{\text{strain}}}{\partial Σ} = \mu (Σ Σ_0^{-1} - I) + \lambda \log\det(Σ Σ_0^{-1}) Σ^{-1}
$$

## 4. 고급 기능

### 4.1 유체 시뮬레이션

**SPH (Smoothed Particle Hydrodynamics) + 가우시안**:

가우시안을 유체 입자로 취급, 공분산을 유체 압력 텐서로 해석:

$$
Σ_i = \frac{k_B T}{p_i} I + \text{Stress}_i
$$

여기서 $p_i$는 밀도, $\text{Stress}_i$는 점성 응력.

**Navier-Stokes 방정식** (가우시안 형태):

$$
\frac{d μ_i}{dt} = v_i, \quad
m_i \frac{d v_i}{dt} = -\nabla p + \mu \nabla^2 v + f_{\text{ext}}
$$

압력 그라디언트는 인접 가우시안 겹침으로 근사.

### 4.2 파괴 (Fracture)

**변형 에너지 임계값**:

$$
\text{If } V_{\text{strain}}(Σ_i) > V_{\text{crit}} \Rightarrow \text{Split } i
$$

가우시안 분할:

```rust
fn split_gaussian(g: GaussianSplat) -> Vec<GaussianSplat> {
    let (U, lambda, _) = eigen_decompose(g.covariance);
    let split_axis = U.column(0);  // 최대 고유벡터
    
    vec![
        GaussianSplat {
            position: g.position + split_axis * 0.5,
            covariance: g.covariance * 0.5,
            mass: g.mass * 0.5,
            ..g
        },
        GaussianSplat {
            position: g.position - split_axis * 0.5,
            covariance: g.covariance * 0.5,
            mass: g.mass * 0.5,
            ..g
        }
    ]
}
```

### 4.3 열 전달

**확산 방정식** (메트릭 기반):

$$
\frac{\partial T_i}{\partial t} = \alpha \sum_j \text{Overlap}(i,j) (T_j - T_i)
$$

온도가 높으면 가우시안이 팽창 ($Σ \propto T$).

### 4.4 소리 합성

가우시안의 진동 → 음파:

$$
p(t) = \sum_i \dot{V}_i(t) = \sum_i \frac{d}{dt} \det(Σ_i)^{1/2}
$$

체적 변화율 = 압력파.

## 5. 벤치마크

### 5.1 성능 목표

| 항목 | 기존 FEM | L-3DGS (목표) |
|------|----------|---------------|
| 해상도 | 10K 정점 | 100K 가우시안 |
| 시뮬레이션 FPS | 30 | 60+ |
| 렌더링 FPS | 60 | 100+ |
| 메모리 (GB) | 2.5 | 1.8 |
| 파괴 처리 | 메시 재생성 | 가우시안 분할 (즉시) |

### 5.2 정확도 검증

**표준 벤치마크**:

1. **탄성체 낙하**: 고무공 → 바닥 충돌 (에너지 보존 확인)
2. **유체 댐 붕괴**: 물 블록 → 중력 낙하 (압축성 확인)
3. **파괴 시나리오**: 유리판 → 총알 충격 (균열 전파)

**비교 대상**:

- Blender (Mantaflow)
- Houdini (Vellum Solver)
- NVIDIA FleX
- PhysX 5.0

## 6. 응용 분야

### 6.1 실시간 게임

- 파괴 가능한 환경 (건물, 지형)
- 유체 효과 (물, 연기, 불)
- 소프트 바디 (천, 고무, 젤리)

**장점**: 메시 재생성 없이 즉시 반응, 렌더링과 물리가 동기화.

### 6.2 VR 수술 시뮬레이터

- 조직 변형 (메스 절개)
- 혈류 시뮬레이션
- 촉각 피드백 (메트릭 변화 → 힘)

### 6.3 디지털 트윈

- 제조 공정 시뮬레이션 (금속 프레스, 사출)
- 재료 물성 추정 (관측 데이터 → 라메 상수 역산)

### 6.4 생성형 3D

- 텍스트 → 3D 모델 + 물리 속성
- "탄력 있는 빨간 공" → 자동으로 적절한 $Σ, \mu, \lambda$ 할당

## 7. 구현 로드맵

### Month 1: 기초 인프라

- Rust 프로젝트 구조 (`reality_stone/applications/l3dgs/`)
- wgpu 렌더링 파이프라인
- 기본 가우시안 래스터화

### Month 2: 물리 엔진 코어

- Symplectic Integrator
- 충돌 감지 (Spatial Hashing)
- 힘 계산 (중력, 스프링)

### Month 3: 변형 및 파괴

- SPD 투영
- 변형 에너지 (Neo-Hookean)
- 가우시안 분할 로직

### Month 4: 고급 효과

- 유체 (SPH 변형)
- 열 전달
- 사운드 합성

### Month 5: 최적화 및 벤치마크

- SIMD 최적화
- GPU 병렬화 (Compute Shader)
- 표준 벤치마크 실행

### Month 6: 데모 및 문서

- Unity/Unreal 플러그인
- 인터랙티브 데모 (Web)
- 논문 작성

## 8. 기술적 도전 과제

### 8.1 수치 안정성

**문제**: SPD 행렬이 ill-conditioned → 고유값 폭발/소실

**해결**:
- 고유값 클램핑 ($\lambda_{\min} < \lambda < \lambda_{\max}$)
- Cholesky 분해 기반 업데이트
- Adaptive time stepping

### 8.2 대규모 장면

**문제**: 100만 가우시안 → 메모리 부족, 느린 충돌 감지

**해결**:
- Octree 기반 LOD
- 멀리 있는 가우시안 병합
- GPU Radix Sort (깊이 정렬)

### 8.3 복잡한 제약 조건

**문제**: 강체 연결, 힌지 조인트 → 제약 위반

**해결**:
- XPBD (Extended Position Based Dynamics)
- Lagrange Multiplier 방법
- Iterative Constraint Solver

## 9. 이론적 기여

### 9.1 새로운 물리 표현

기존: 물체 = 메시 (정점 + 삼각형)
제안: 물체 = 가우시안 집합 (위치 + 메트릭)

**장점**:
- 위상 변화 자연스러움 (파괴, 합체)
- 해상도 독립적 (확대해도 부드러움)
- 렌더링과 시뮬레이션 통합

### 9.2 리만 기하학과 컴퓨터 그래픽스 연결

메트릭 텐서를 직접 시뮬레이션 → 일반 상대론의 물리학을 실시간 그래픽스에 적용.

**응용 가능성**:
- 블랙홀 렌즈 효과
- 시공간 왜곡 (워프 드라이브)
- 비유클리드 공간 게임 (HyperRogue 스타일)

## 10. 결론

L-3DGS는 Reality Stone의 라그랑지안-리만 프레임워크를 3D 그래픽스에 적용하여:

1. 렌더링과 물리를 단일 자료구조로 통합
2. 메시 없는 변형, 파괴, 유체 시뮬레이션
3. 실시간 성능 (60+ FPS)
4. 새로운 표현 공간 (가우시안 = 메트릭)

이는 차세대 게임 엔진과 시뮬레이터의 기반이 될 수 있습니다.

