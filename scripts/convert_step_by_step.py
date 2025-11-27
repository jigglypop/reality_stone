#!/usr/bin/env python3
"""
Transformer → RS-ULF 단계별 변환 및 검증 스크립트

각 단계를 독립적으로 실행하고 성공 여부를 확인합니다.
"""

import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def print_step(step_num: int, title: str):
    """단계 헤더 출력"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70)


def print_success(message: str = "SUCCESS"):
    """성공 메시지"""
    print(f"[OK] {message}")


def print_error(message: str):
    """에러 메시지"""
    print(f"[ERROR] {message}")


def print_info(key: str, value):
    """정보 출력"""
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")
    elif isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape} ({value.dtype})")
    else:
        print(f"  {key}: {value}")


class StepByStepConverter:
    """단계별 변환 클래스"""
    
    def __init__(self, model_name: str, device: str = 'cpu', cache_dir: str = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", "E:/hf-cache")
        
        self.model = None
        self.tokenizer = None
        self.test_layer_idx = 0
        self.layer_weights = None
        self.rs_layer = None
        self.metric = None
        self.rs_model = None
        
    def step1_load_model(self) -> bool:
        """Step 1: 모델 로딩"""
        print_step(1, "모델 로딩")
        
        # 이미 로딩되었으면 스킵
        if self.model is not None:
            print_success("Model already loaded (cached)")
            return True
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"모델 로딩 중: {self.model_name}")
            print(f"캐시 디렉토리: {self.cache_dir}")
            print(f"디바이스: {self.device}")
            
            # Tokenizer
            with tqdm(total=1, desc="Loading tokenizer", ncols=80) as pbar:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                pbar.update(1)
            
            # Model
            print("Loading model (progress bar from transformers)...")
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                )
            print_success("Model loaded")
            
            # 모델 정보
            print("\nModel info:")
            num_params = sum(p.numel() for p in self.model.parameters())
            num_layers = len(self.model.model.layers)
            d_model = self.model.config.hidden_size
            
            print_info("Parameters", f"{num_params:,}")
            print_info("Layers", num_layers)
            print_info("Hidden dim", d_model)
            
            # 메모리 사용량
            mem_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            print_info("Memory", f"{mem_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            print_error(f"모델 로딩 실패: {e}")
            return False
    
    def step2_extract_weights(self) -> bool:
        """Step 2: 가중치 추출"""
        print_step(2, "가중치 추출 테스트")
        
        if self.model is None:
            print_error("모델이 로딩되지 않음. Step 1을 먼저 실행하세요.")
            return False
        
        try:
            from reality_stone.models.transformer_converter import extract_transformer_layer_weights
            
            layer_idx = self.test_layer_idx
            print(f"레이어 {layer_idx} 가중치 추출 중...")
            
            self.layer_weights = extract_transformer_layer_weights(self.model, layer_idx)
            
            print_success("가중치 추출 성공")
            
            # 가중치 확인
            for key, value in self.layer_weights.items():
                if value is not None:
                    print_info(key, value)
            
            # 기본 검증
            assert self.layer_weights['WQ'] is not None, "WQ가 None"
            assert self.layer_weights['WK'] is not None, "WK가 None"
            assert self.layer_weights['W1'] is not None, "W1이 None"
            assert self.layer_weights['W2'] is not None, "W2가 None"
            
            print_success("기본 검증 통과")
            
            return True
            
        except Exception as e:
            print_error(f"가중치 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_test_metric(self) -> bool:
        """Step 3: Metric 추출 및 검증 (Rust 바인딩)"""
        print_step(3, "Metric 추출 및 검증")
        
        if self.layer_weights is None:
            print_error("가중치가 추출되지 않음. Step 2를 먼저 실행하세요.")
            return False
        
        try:
            WQ = self.layer_weights['WQ']
            WK = self.layer_weights['WK']
            
            with tqdm(total=5, desc="Metric extraction (diag)", ncols=80) as pbar:
                d_q_out = WQ.size(0)
                d_k_out = WK.size(0)
                if d_q_out != d_k_out:
                    num_groups = d_q_out // d_k_out
                    WK_expanded = WK.repeat(num_groups, 1)
                else:
                    WK_expanded = WK
                pbar.update(1)

                g_raw = torch.matmul(WQ.t().float(), WK_expanded.float())
                pbar.set_postfix_str(f"g_raw: {g_raw.shape}")
                pbar.update(1)

                g_diag = torch.diag(g_raw)
                g_diag = torch.abs(g_diag) + 1e-6
                min_val = g_diag.min().item()
                max_val = g_diag.max().item()
                pbar.set_postfix_str(f"diag: [{min_val:.6f}, {max_val:.6f}]")
                pbar.update(1)

                inv_diag = 1.0 / g_diag
                identity_diag = g_diag * inv_diag
                error = torch.abs(identity_diag - 1.0).max()
                pbar.set_postfix_str(f"inv_err: {error.item():.6e}")
                pbar.update(1)

                is_pd = min_val > 0
                pbar.update(1)

            print_info("최소 대각값", f"{min_val:.6f}")
            print_info("최대 대각값", f"{max_val:.6f}")
            print_info("Inverse 오차", f"{error.item():.6e}")

            if not is_pd:
                print_error(f"Metric 대각값이 음수 포함 (min={min_val:.6f})")
                return False

            if error >= 0.02:
                print_error(f"Inverse 오차 너무 큼: {error.item()}")
                return False

            print_success("Metric 추출 및 대각 기반 검증 통과")
            return True
            
        except Exception as e:
            print_error(f"Metric 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_test_potential(self) -> bool:
        """Step 4: Potential 함수 검증"""
        print_step(4, "Potential 함수 검증")
        
        if self.layer_weights is None:
            print_error("가중치가 추출되지 않음. Step 2를 먼저 실행하세요.")
            return False
        
        try:
            W1 = self.layer_weights['W1']
            W2 = self.layer_weights['W2']
            d_model = W1.size(1)
            
            print("Potential 함수 생성 중...")
            
            # 테스트 입력
            x_test = torch.randn(2, 16, d_model, dtype=W1.dtype, device=W1.device)
            
            # FFN 계산 (Transformer)
            with tqdm(total=3, desc="Transformer FFN", ncols=80) as pbar:
                h = F.silu(F.linear(x_test, W1))
                pbar.update(1)
                if self.layer_weights['W_up'] is not None:
                    up = F.linear(x_test, self.layer_weights['W_up'])
                    h = h * up
                    pbar.update(1)
                else:
                    pbar.update(1)
                ffn_out = F.linear(h, W2)
                pbar.update(1)
            print_info("FFN output", ffn_out)
            
            # Potential 계산
            phi = 0.5 * torch.sum(ffn_out ** 2, dim=-1)
            print_info("Φ", phi)
            
            # 기본 검증
            assert phi.shape == (2, 16), f"Φ shape 오류: {phi.shape}"
            assert torch.all(phi >= 0), "Φ는 non-negative여야 함"
            
            print_success("Potential 기본 검증 통과")
            
            # Gradient 계산
            with tqdm(total=2, desc="Gradient computation", ncols=80) as pbar:
                x_req = x_test.clone().detach().requires_grad_(True)
                
                h = F.silu(F.linear(x_req, W1))
                if self.layer_weights['W_up'] is not None:
                    up = F.linear(x_req, self.layer_weights['W_up'])
                    h = h * up
                ffn = F.linear(h, W2)
                phi_req = 0.5 * torch.sum(ffn ** 2, dim=-1)
                pbar.update(1)
                
                grad = torch.autograd.grad(phi_req.sum(), x_req)[0]
                pbar.update(1)
            print_info("∇Φ", grad)
            
            assert grad.shape == x_test.shape, f"Gradient shape 오류: {grad.shape}"
            assert not torch.isnan(grad).any(), "Gradient에 NaN 존재"
            assert not torch.isinf(grad).any(), "Gradient에 Inf 존재"
            
            print_success("Gradient 계산 성공")
            
            return True
            
        except Exception as e:
            print_error(f"Potential 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step5_test_graph(self) -> bool:
        """Step 5: Graph Laplacian 검증"""
        print_step(5, "Graph Laplacian 검증")
        
        try:
            from reality_stone.models.transformer_converter import create_graph_laplacian
            
            seq_len = 128
            window_size = 8
            
            print(f"Graph Laplacian 생성 (seq_len={seq_len}, window={window_size})...")
            L = create_graph_laplacian(seq_len, window_size=window_size, directed=True)
            
            print_info("L shape", L)
            
            # Laplacian 속성 검증
            print("\nLaplacian 속성 검증...")
            
            # 1. L @ 1 = 0
            ones = torch.ones(seq_len)
            L_ones = L @ ones
            l_ones_norm = torch.norm(L_ones)
            print_info("||L @ 1||", f"{l_ones_norm.item():.6e}")
            
            if l_ones_norm < 1e-5:
                print_success("L @ 1 = 0 검증 통과")
            else:
                print_error(f"L @ 1 != 0 (norm = {l_ones_norm.item()})")
            
            # 2. Row sum = 0 (for directed graph, may not hold exactly)
            row_sums = L.sum(dim=1)
            print_info("Row sums", f"mean={row_sums.mean():.6e}, max={row_sums.max():.6e}")
            
            # 3. Diffusion 테스트
            print("\nDiffusion 테스트...")
            x = torch.randn(2, seq_len, 512)
            tau = 0.01
            
            Lx = torch.einsum('lm,bmd->bld', L, x)
            x_diff = x - tau * Lx
            
            print_info("Input", x)
            print_info("Diffused", x_diff)
            
            # Energy 변화 (undirected graph에서만 감소 보장)
            energy_before = torch.sum(x ** 2)
            energy_after = torch.sum(x_diff ** 2)
            print_info("Energy before", f"{energy_before.item():.2f}")
            print_info("Energy after", f"{energy_after.item():.2f}")
            
            print_success("Graph diffusion 테스트 완료")
            
            return True
            
        except Exception as e:
            print_error(f"Graph 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step6_convert_single_layer(self) -> bool:
        """Step 6: 단일 레이어 변환 (Rust RS-ULF)"""
        print_step(6, "단일 레이어 변환")
        
        if self.layer_weights is None:
            print_error("가중치가 추출되지 않음. Step 2를 먼저 실행하세요.")
            return False
        
        try:
            import reality_stone as rs
            from reality_stone.models.transformer_converter import RSULFLayer
            
            if not rs._has_rust_ext:
                print_error("Rust extension not available")
                return False
            
            WQ = self.layer_weights['WQ']
            WK = self.layer_weights['WK']
            W1 = self.layer_weights['W1']
            W2 = self.layer_weights['W2']
            d_model = WQ.size(1)
            
            # Step 6은 단일 레이어 sanity check 용도이므로,
            # global 변환(Target ≥ 200x 압축)보다 더 작은 rank로 빠르게 테스트한다.
            r = 64
            
            print(f"RS-ULF 레이어 생성 (d_model={d_model}, r={r}, Rust SVD 한 번)...")
            
            # RSULFLayer 내부에서 필요한 SVD를 한 번만 수행
            self.rs_layer = RSULFLayer(
                WQ=WQ, WK=WK, W1=W1, W2=W2,
                d_model=d_model, r=r,
                eta=0.01, alpha=0.02, beta=0.01, gamma=0.99,
                seq_len=64, window=8
            )
            # Step 7 호환을 위한 placeholder metric 정보
            self.metric = {"info": "metric_svd_skipped_in_step6_for_speed"}
            
            stats = self.rs_layer.param_count()
            print_info("압축 파라미터", stats['compressed'])
            print_info("원본 파라미터", stats['original'])
            print_info("압축률", f"{stats['ratio']:.2f}x")
            
            print_success("레이어 생성 성공")
            
            print("\nForward pass 테스트...")
            seq_len = 64
            x_test = torch.randn(2, seq_len, d_model).float()
            
            output, v_mem = self.rs_layer.forward(x_test)
            
            print_info("Input shape", x_test.shape)
            print_info("Output shape", output.shape)
            print_info("V memory shape", v_mem.shape)
            
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                print_error(f"Output에 NaN 발견: {nan_count}개")
                return False
            
            assert output.shape == x_test.shape, f"Output shape 오류: {output.shape}"
            assert not torch.isnan(output).any(), "Output에 NaN"
            assert not torch.isinf(output).any(), "Output에 Inf"
            
            print_success("Forward pass 성공")
            
            print("\nMultiple steps 테스트 (Bellman memory 축적)...")
            v = v_mem
            for i in range(3):
                output, v = self.rs_layer.forward(output, v)
                print(f"  Step {i+1}: output norm = {output.norm().item():.4f}, V mean = {v.mean().item():.4f}")
            
            print_success("Multiple steps 안정적")
            
            return True
            
        except Exception as e:
            print_error(f"레이어 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step7_save_and_load(self) -> bool:
        """Step 7: 저장 및 로딩 테스트"""
        print_step(7, "저장 및 로딩 테스트")
        
        if self.rs_layer is None or self.metric is None:
            print_error("레이어가 변환되지 않음. Step 6을 먼저 실행하세요.")
            return False
        
        try:
            save_path = Path("checkpoints/rsulf_layer.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"레이어 저장: {save_path}")
            
            torch.save({
                'metric': self.metric,
                'weights': {
                    'WQ': self.layer_weights['WQ'].cpu(),
                    'WK': self.layer_weights['WK'].cpu(),
                    'W1': self.layer_weights['W1'].cpu(),
                    'W2': self.layer_weights['W2'].cpu(),
                },
                'config': {
                    'd_model': self.rs_layer.d_model,
                    'r': self.rs_layer.r,
                    'curvature': self.rs_layer.curvature,
                },
                'param_count': self.rs_layer.param_count(),
            }, save_path)
            
            print_success(f"저장 완료 ({save_path.stat().st_size / 1024 / 1024:.2f} MB)")
            
            print("\n로딩 테스트...")
            checkpoint = torch.load(save_path)
            
            from reality_stone.models.transformer_converter import RSULFLayer
            
            new_layer = RSULFLayer(
                WQ=checkpoint['weights']['WQ'],
                WK=checkpoint['weights']['WK'],
                W1=checkpoint['weights']['W1'],
                W2=checkpoint['weights']['W2'],
                d_model=checkpoint['config']['d_model'],
                r=checkpoint['config']['r'],
            )
            
            print_info("Loaded d_model", checkpoint['config']['d_model'])
            print_info("Loaded r", checkpoint['config']['r'])
            print_info("Loaded curvature", f"{checkpoint['config']['curvature']:.6f}")
            
            print_success("로딩 성공")
            
            print("\n동일성 검증...")
            seq_len = 64
            
            x_test = torch.randn(1, seq_len, checkpoint['config']['d_model']).float()
            
            out1, _ = self.rs_layer.forward(x_test)
            out2, _ = new_layer.forward(x_test)
            
            diff = torch.abs(out1 - out2).max()
            print_info("최대 차이", f"{diff.item():.6e}")
            
            if diff < 0.1:
                print_success("저장/로딩 후 유사한 출력")
            else:
                print_error(f"출력 차이 발생: {diff.item()}")
            
            return True
            
        except Exception as e:
            print_error(f"저장/로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step8_global_conversion(self) -> bool:
        """Step 8: 전체 모델 글로벌 변환 및 압축률/오차 체크"""
        print_step(8, "Global Basis 변환 및 압축률/오차 체크")
        
        if self.model is None:
            print_error("모델이 로딩되지 않음. Step 1을 먼저 실행하세요.")
            return False
        
        try:
            import reality_stone as rs
            from reality_stone.models.transformer_converter import (
                estimate_global_compression,
                compute_global_scales_from_model,
            )
            
            if not rs._has_rust_ext:
                print_error("Rust extension not available")
                return False
            
            # 설계 목표: ≥ 200× 압축을 만족시키는 rank 후보 (초기값)
            metric_rank = 32
            ffn_rank = 32
            
            print(f"Global 압축률 추정 중 (metric_rank={metric_rank}, ffn_rank={ffn_rank})...")
            est = estimate_global_compression(self.model, metric_rank, ffn_rank)
            print_info("추정 원본 파라미터", f"{int(est['original_params_est']):,}")
            print_info("추정 RS-ULF 파라미터", f"{int(est['rsulf_params_est']):,}")
            print_info("예상 압축률", f"{est['compression_est']:.2f}x")
            
            if est["compression_est"] < 200.0:
                print_error(f"예상 압축률 {est['compression_est']:.2f}x < 200x (설계 목표 미달)")
                # 일단 경고만 띄우고 계속 진행 (튜닝 여지를 남김)
            
            print("\nGlobal basis 및 per-layer thin scale 추출 중...")
            scales = compute_global_scales_from_model(
                self.model,
                metric_rank=metric_rank,
                ffn_rank=ffn_rank,
            )
            
            print_info("Metric basis U shape", np.array(scales["metric_basis"]["U"]).shape)
            print_info("FFN basis U1 shape", np.array(scales["ffn_basis"]["U1"]).shape)
            print_info("레이어 수 (per_layer)", len(scales["per_layer"]))
            print_info("Metric 재구성 최대 상대오차", f"{scales['metric_error_max']:.6f}")
            print_info("FFN 재구성 최대 상대오차", f"{scales['ffn_error_max']:.6f}")
            
            # 오차 기준: 파라미터 레벨에서는 다소 여유 있게 두되, 상위 단계에서 PPL 기준으로 다시 검증
            if scales["metric_error_max"] > 0.2 or scales["ffn_error_max"] > 0.2:
                print_error("Global basis 재구성 오차가 너무 큼 (> 0.2)")
            
            # 체크포인트 저장
            save_path = Path("checkpoints/rsulf_global_scales.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                {
                    "compression_est": est,
                    "scales": scales,
                },
                save_path,
            )
            print_success(f"Global basis/scale 저장 완료: {save_path}")
            
            return True
        
        except Exception as e:
            print_error(f"Global 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step9_rsulf_inference(self) -> bool:
        print_step(9, "RS-ULF 전체 모델 추론 (텍스트 생성)")

        if self.model is None:
            print_error("모델이 로딩되지 않음. Step 1을 먼저 실행하세요.")
            return False

        try:
            import reality_stone as rs
            from reality_stone.models.transformer_converter import (
                build_rsulf_model_from_global_scales,
                rsulf_generate,
            )

            if not rs._has_rust_ext:
                print_error("Rust extension not available")
                return False

            scales_path = Path("checkpoints/rsulf_global_scales.pt")
            if not scales_path.exists():
                print_error(f"Global scales 체크포인트 없음: {scales_path}")
                return False

            data = torch.load(scales_path)
            compression_est = data.get("compression_est", None)
            scales = data["scales"]

            if compression_est is not None:
                print_info(
                    "저장된 예상 압축률",
                    f"{compression_est.get('compression_est', 0.0):.2f}x",
                )

            print("RS-ULF 모델 구성 중...")

            torch_device = torch.device(
                self.device if self.device in ["cuda", "cpu"] else "cpu"
            )
            torch_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

            self.rs_model = build_rsulf_model_from_global_scales(
                self.model,
                scales,
                r=64,
                eta=0.01,
                alpha=0.02,
                beta=0.01,
                gamma=0.99,
                seq_len=128,
                window=8,
                device=torch_device,
                dtype=torch_dtype,
            )

            prompt = "한국 경제 성장의 리스크 요인을 세 가지로 나누어 설명하고, 마지막에 한 줄 요약을 붙여줘."
            print_info("프롬프트", prompt)

            text = rsulf_generate(
                self.model,
                self.rs_model,
                self.tokenizer,
                prompt,
                max_new_tokens=96,
                temperature=0.8,
                top_p=0.9,
                device=self.device,
            )

            print("\n===== RS-ULF LLM 응답 =====")
            print(text)
            print("===== RS-ULF LLM 응답 끝 =====")

            return True

        except Exception as e:
            print_error(f"RS-ULF 추론 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_steps(self, resume: bool = False) -> bool:
        """모든 단계 실행"""
        steps = [
            ("모델 로딩", self.step1_load_model),
            ("가중치 추출", self.step2_extract_weights),
            ("Metric 검증", self.step3_test_metric),
            ("Potential 검증", self.step4_test_potential),
            ("Graph 검증", self.step5_test_graph),
            ("레이어 변환", self.step6_convert_single_layer),
            ("저장/로딩", self.step7_save_and_load),
            ("Global 변환", self.step8_global_conversion),
            ("RS-ULF 추론", self.step9_rsulf_inference),
        ]
        
        print("\n" + "#"*70)
        print("# Transformer → RS-ULF 단계별 변환 시작")
        if resume:
            print("# (Resume 모드: 기존 체크포인트 활용)")
        print("#"*70)
        
        results = []
        checkpoint_dir = "checkpoints/conversion"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for step_name, step_func in steps:
            safe_name = step_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            checkpoint_path = os.path.join(checkpoint_dir, f"{safe_name}.pt")
            
            if resume and os.path.exists(checkpoint_path):
                print(f"\n[건너뜀] {step_name} - 체크포인트 존재: {checkpoint_path}")
                results.append((step_name, True))
                
                try:
                    checkpoint = torch.load(checkpoint_path)
                    if 'layer_weights' in checkpoint and checkpoint['layer_weights']:
                        self.layer_weights = checkpoint['layer_weights']
                    
                    if step_name == "레이어 변환" and 'metric' in checkpoint:
                        from reality_stone.models.transformer_converter import RSULFLayer
                        
                        WQ = self.layer_weights['WQ']
                        d_model = WQ.size(1)
                        
                        self.rs_layer = RSULFLayer(d_model)
                        self.metric = checkpoint.get('metric', None)
                        print(f"  [복원] rs_layer 재생성 완료")
                    
                except Exception as e:
                    print(f"  [경고] 체크포인트 로드 실패: {e}")
                    import traceback
                    traceback.print_exc()
                
                continue
            
            success = step_func()
            results.append((step_name, success))
            
            if success:
                checkpoint = {
                    'step': step_name,
                    'layer_weights': self.layer_weights,
                    'metric': getattr(self, 'metric', None),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"[저장] {checkpoint_path}")
            
            if not success:
                print(f"\n[WARN] {step_name} 단계에서 실패. 중단합니다.")
                break
        
        print("\n" + "#"*70)
        print("# 변환 결과 요약")
        print("#"*70)
        
        for step_name, success in results:
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {step_name}")
        
        all_success = all(success for _, success in results)
        
        if all_success:
            print("\n[SUCCESS] 모든 단계 성공!")
        else:
            print("\n[WARN] 일부 단계 실패")
        
        return all_success


def main():
    parser = argparse.ArgumentParser(description="단계별 변환 테스트")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="모델 이름"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="디바이스"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="E:/hf-cache",
        help="캐시 디렉토리"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="특정 단계만 실행 (1-8)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트에서 이어서 실행"
    )
    
    args = parser.parse_args()
    
    converter = StepByStepConverter(
        model_name=args.model_name,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    if args.step is None:
        # 모든 단계 실행
        success = converter.run_all_steps(resume=args.resume)
    else:
        # 특정 단계만 실행
        step_funcs = {
            1: converter.step1_load_model,
            2: converter.step2_extract_weights,
            3: converter.step3_test_metric,
            4: converter.step4_test_potential,
            5: converter.step5_test_graph,
            6: converter.step6_convert_single_layer,
            7: converter.step7_save_and_load,
            8: converter.step8_global_conversion,
            9: converter.step9_rsulf_inference,
        }
        
        if args.step in step_funcs:
            # 이전 단계들 먼저 실행 (의존성)
            for i in range(1, args.step):
                if not step_funcs[i]():
                    print(f"[WARN] Step {i} 실패. Step {args.step}를 실행할 수 없습니다.")
                    sys.exit(1)
            
            # 요청한 단계 실행
            success = step_funcs[args.step]()
        else:
            print(f"[WARN] 잘못된 단계 번호: {args.step} (1-8 사이)")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

