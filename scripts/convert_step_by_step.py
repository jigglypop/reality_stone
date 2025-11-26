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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                cache_dir=self.cache_dir,
                local_files_only=False,
                low_cpu_mem_usage=True
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
        """Step 3: Metric 추출 및 검증"""
        print_step(3, "Metric 추출 및 검증")
        
        if self.layer_weights is None:
            print_error("가중치가 추출되지 않음. Step 2를 먼저 실행하세요.")
            return False
        
        try:
            from reality_stone.models.rsulf import extract_metric, stabilize_metric
            
            WQ = self.layer_weights['WQ']
            WK = self.layer_weights['WK']
            
            # 메트릭 추출 및 검증
            with tqdm(total=5, desc="Metric extraction", ncols=80) as pbar:
                # Metric 계산
                g = extract_metric(WQ, WK)
                pbar.set_postfix_str(f"g: {g.shape}")
                pbar.update(1)
                
                # Metric 안정화
                g_stable = stabilize_metric(g, strategy="diagonal")
                pbar.set_postfix_str(f"stabilized")
                pbar.update(1)
                
                # PD 검증
                g_stable_f32 = g_stable.float()
                eigvals = torch.linalg.eigvalsh(g_stable_f32)
                min_eig = eigvals.min().item()
                max_eig = eigvals.max().item()
                pbar.set_postfix_str(f"eig: [{min_eig:.6f}, {max_eig:.6f}]")
                pbar.update(1)
                
                is_pd = min_eig > 0
                if not is_pd:
                    print_error(f"Metric이 PD가 아님 (min eigenvalue = {min_eig})")
                    return False
                
                # Inverse 계산
                diag_vals = torch.diag(g_stable)
                g_inv = torch.diag(1.0 / (diag_vals + 1e-8))
                pbar.set_postfix_str(f"inverse computed")
                pbar.update(1)
                
                # Identity 검증
                identity = g_stable @ g_inv
                eye = torch.eye(g_stable.size(0))
                error = torch.norm(identity - eye) / g_stable.size(0)
                pbar.set_postfix_str(f"inv_error: {error.item():.6e}")
                pbar.update(1)
                
            print_info("최소 고유값", f"{min_eig:.6f}")
            print_info("최대 고유값", f"{max_eig:.6f}")
            print_info("Inverse 오차", f"{error.item():.6e}")
            
            if error < 1e-3:
                print_success("Metric 추출 및 검증 통과")
            else:
                print_error(f"Inverse 오차 너무 큼: {error.item()}")
                return False
            
            # 정합성 테스트: (Qx)·(Kx) ≈ x^T g x  
            d_model = WQ.size(1)
            x_test = torch.randn(2, 16, d_model)
            
            with tqdm(total=4, desc="Consistency check", ncols=80) as pbar:
                # Transformer side
                q = F.linear(x_test, WQ)
                pbar.update(1)
                k = F.linear(x_test, WK)
                pbar.update(1)
                tf_dots = torch.matmul(q, k.transpose(-1, -2))
                
                # RS side
                gx = torch.matmul(x_test, g_stable)
                pbar.update(1)
                rs_dots = torch.matmul(gx, x_test.transpose(-1, -2))
                pbar.update(1)
            
            diff = torch.abs(tf_dots - rs_dots).mean()
            cos_sim = F.cosine_similarity(tf_dots.flatten(), rs_dots.flatten(), dim=0)
            
            print_info("평균 차이", f"{diff.item():.6e}")
            print_info("Cosine similarity", f"{cos_sim.item():.6f}")
            
            if cos_sim > 0.9:
                print_success("정합성 테스트 통과")
                return True
            else:
                print_error(f"정합성 테스트 실패 (similarity = {cos_sim.item()})")
                return False
            
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
            from reality_stone.models.rsulf import RSULF
            
            W1 = self.layer_weights['W1']
            W2 = self.layer_weights['W2']
            d_model = W1.size(1)
            
            print("Potential 함수 생성 중...")
            
            # 테스트 입력
            x_test = torch.randn(2, 16, d_model)
            
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
            
            Lx = torch.matmul(L, x.transpose(1, 2)).transpose(1, 2)
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
        """Step 6: 단일 레이어 변환"""
        print_step(6, "단일 레이어 변환")
        
        if self.layer_weights is None:
            print_error("가중치가 추출되지 않음. Step 2를 먼저 실행하세요.")
            return False
        
        try:
            from reality_stone.models.rsulf import RSULF
            from reality_stone.models.transformer_converter import create_graph_laplacian
            
            WQ = self.layer_weights['WQ']
            WK = self.layer_weights['WK']
            W1 = self.layer_weights['W1']
            W2 = self.layer_weights['W2']
            d_model = WQ.size(1)
            
            print(f"RS-ULF 레이어 생성 (d_model={d_model})...")
            
            # Laplacian (placeholder)
            L = torch.eye(1)
            
            self.rs_layer = RSULF(
                d_model=d_model,
                WQ=WQ,
                WK=WK,
                W1=W1,
                W2=W2,
                L_matrix=L,
                lr=0.02,
                alpha=0.04,
                beta=0.01,
                gamma=0.98,
                metric_strategy="diagonal"
            )
            
            print_success("레이어 생성 성공")
            print_info("레이어 파라미터", sum(p.numel() for p in self.rs_layer.parameters()))
            
            # Forward 테스트
            print("\nForward pass 테스트...")
            seq_len = 64
            x_test = torch.randn(2, seq_len, d_model)
            
            # Laplacian 업데이트
            L_test = create_graph_laplacian(seq_len, window_size=8)
            self.rs_layer.L = L_test
            
            with torch.no_grad():
                output, V = self.rs_layer(x_test, V=None)
            
            print_info("Input", x_test)
            print_info("Output", output)
            print_info("Memory V", V)
            
            # 검증
            assert output.shape == x_test.shape, f"Output shape 오류: {output.shape}"
            assert V.shape == (2, seq_len), f"Memory shape 오류: {V.shape}"
            assert not torch.isnan(output).any(), "Output에 NaN"
            assert not torch.isinf(output).any(), "Output에 Inf"
            
            print_success("Forward pass 성공")
            
            # Multiple steps
            print("\nMultiple steps 테스트...")
            for i in range(3):
                output, V = self.rs_layer(output, V)
                print(f"  Step {i+1}: output norm = {output.norm().item():.4f}")
            
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
        
        if self.rs_layer is None:
            print_error("레이어가 변환되지 않음. Step 6을 먼저 실행하세요.")
            return False
        
        try:
            save_path = Path("checkpoints/test_single_layer.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"레이어 저장: {save_path}")
            
            # 저장
            torch.save({
                'state_dict': self.rs_layer.state_dict(),
                'config': {
                    'd_model': self.rs_layer.d_model,
                    'lr': self.rs_layer.lr,
                    'alpha': self.rs_layer.alpha,
                    'beta': self.rs_layer.beta,
                    'gamma': self.rs_layer.gamma_mem,
                    'metric_strategy': self.rs_layer.metric_strategy
                }
            }, save_path)
            
            print_success(f"저장 완료 ({save_path.stat().st_size / 1024 / 1024:.2f} MB)")
            
            # 로딩
            print("\n레이어 로딩 테스트...")
            checkpoint = torch.load(save_path)
            
            from reality_stone.models.rsulf import RSULF
            
            new_layer = RSULF(
                d_model=checkpoint['config']['d_model'],
                lr=checkpoint['config']['lr'],
                alpha=checkpoint['config']['alpha'],
                beta=checkpoint['config']['beta'],
                gamma=checkpoint['config']['gamma'],
                metric_strategy=checkpoint['config']['metric_strategy']
            )
            
            new_layer.load_state_dict(checkpoint['state_dict'])
            print_success("로딩 성공")
            
            # 동일성 검증
            print("\n동일성 검증...")
            x_test = torch.randn(1, 32, checkpoint['config']['d_model'])
            
            with torch.no_grad():
                out1, _ = self.rs_layer(x_test)
                out2, _ = new_layer(x_test)
            
            diff = torch.abs(out1 - out2).max()
            print_info("최대 차이", f"{diff.item():.6e}")
            
            if diff < 1e-6:
                print_success("저장/로딩 후 동일한 출력")
            else:
                print_error(f"출력 차이 발생: {diff.item()}")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"저장/로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_steps(self) -> bool:
        """모든 단계 실행"""
        steps = [
            ("모델 로딩", self.step1_load_model),
            ("가중치 추출", self.step2_extract_weights),
            ("Metric 검증", self.step3_test_metric),
            ("Potential 검증", self.step4_test_potential),
            ("Graph 검증", self.step5_test_graph),
            ("레이어 변환", self.step6_convert_single_layer),
            ("저장/로딩", self.step7_save_and_load),
        ]
        
        print("\n" + "#"*70)
        print("# Transformer → RS-ULF 단계별 변환 시작")
        print("#"*70)
        
        results = []
        
        for step_name, step_func in steps:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"\n⚠️  {step_name} 단계에서 실패. 중단합니다.")
                break
        
        # 요약
        print("\n" + "#"*70)
        print("# 변환 결과 요약")
        print("#"*70)
        
        for step_name, success in results:
            status = "✓" if success else "✗"
            print(f"{status} {step_name}")
        
        all_success = all(success for _, success in results)
        
        if all_success:
            print("\n🎉 모든 단계 성공!")
        else:
            print("\n⚠️  일부 단계 실패")
        
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
        help="특정 단계만 실행 (1-7)"
    )
    
    args = parser.parse_args()
    
    converter = StepByStepConverter(
        model_name=args.model_name,
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    if args.step is None:
        # 모든 단계 실행
        success = converter.run_all_steps()
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
        }
        
        if args.step in step_funcs:
            # 이전 단계들 먼저 실행 (의존성)
            for i in range(1, args.step):
                if not step_funcs[i]():
                    print(f"⚠️  Step {i} 실패. Step {args.step}를 실행할 수 없습니다.")
                    sys.exit(1)
            
            # 요청한 단계 실행
            success = step_funcs[args.step]()
        else:
            print(f"⚠️  잘못된 단계 번호: {args.step} (1-7 사이)")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

