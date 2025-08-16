# 05. Security — Key and Metric Handling

- Key in HSM/TEE: G_k = L_k^T L_k 생성·적용을 TEE 내부에서
- Session Rotation: L~_k = R_s L_k (요청별 임시 회전)
- Logs: 점수/거리 로그 최소화·양자화·노이즈·쿼리율 제한
- Exposure: 변환 임베딩을 디스크 상주 금지, on-the-fly 메모리 캐시
- Collision/Ambiguity: 고유값 패턴에 키 비트 혼합, 정규형 규칙 채택
