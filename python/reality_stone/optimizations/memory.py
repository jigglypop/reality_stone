import torch
import torch.nn as nn
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
from contextlib import contextmanager

@dataclass
class OptimizationConfig:
    use_memory_efficient: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    use_cuda_if_available: bool = True
    cuda_memory_fraction: float = 0.8
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    # Fused 연산 최적화
    prefer_fused_ops: bool = True
    fused_threshold_size: int = 1000  
    use_torch_compile: bool = False
    compile_mode: str = "default"  
    adaptive_batch_size: bool = False
    max_batch_size: int = 512
    min_batch_size: int = 32

class PerformanceProfiler:
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[int]] = {}
        self.enabled = True
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    @contextmanager
    def profile(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        start_time = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_used = end_memory - start_memory
            else:
                memory_used = 0
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000  # ms
            if name not in self.timings:
                self.timings[name] = []
                self.memory_usage[name] = []
            self.timings[name].append(elapsed)
            self.memory_usage[name].append(memory_used)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        if name not in self.timings:
            return {}
        timings = self.timings[name]
        memory = self.memory_usage[name]
        return {
            'avg_time_ms': sum(timings) / len(timings),
            'min_time_ms': min(timings),
            'max_time_ms': max(timings),
            'total_calls': len(timings),
            'avg_memory_bytes': sum(memory) / len(memory) if memory else 0,
            'max_memory_bytes': max(memory) if memory else 0
        }
    
    def print_summary(self):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("Reality Stone Performance Summary")
        print("="*60)
        
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Average Time: {stats['avg_time_ms']:.3f} ms")
            print(f"  Total Calls:  {stats['total_calls']}")
            if stats['avg_memory_bytes'] > 0:
                print(f"  Avg Memory:   {stats['avg_memory_bytes'] / 1024 / 1024:.2f} MB")
        print("="*60)

# 전역 프로파일러
global_profiler = PerformanceProfiler()

def enable_profiling():
    """성능 프로파일링 활성화"""
    global_profiler.enable()

def disable_profiling():
    """성능 프로파일링 비활성화"""  
    global_profiler.disable()

def print_performance_summary():
    """성능 요약 출력"""
    global_profiler.print_summary()

def setup_optimizations(config: OptimizationConfig):
    """최적화 설정 적용"""
    
    # CUDA 설정
    if config.use_cuda_if_available and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
        
        # 메모리 분할 설정
        if config.cuda_memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(config.cuda_memory_fraction)
    
    # Mixed Precision 설정
    if config.mixed_precision:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            warnings.warn("TF32 optimization not available")

class OptimizedModel(nn.Module):
    """최적화가 적용된 모델 래퍼"""
    
    def __init__(self, 
                 model: nn.Module, 
                 config: OptimizationConfig = None):
        super().__init__()
        self.model = model
        self.config = config or OptimizationConfig()
        self.compiled_model = None
        
        self._apply_optimizations()
        
    def _apply_optimizations(self):
        """최적화 적용"""
        
        # PyTorch 2.0 컴파일 (선택적)
        if self.config.use_torch_compile:
            try:
                self.compiled_model = torch.compile(
                    self.model, 
                    mode=self.config.compile_mode
                )
                print(f"Model compiled with mode: {self.config.compile_mode}")
            except Exception as e:
                warnings.warn(f"Compilation failed: {e}")
                self.compiled_model = None
        
        # 메모리 효율성 설정
        if self.config.use_memory_efficient:
            # 그래디언트 체크포인팅 (선택적)
            if self.config.gradient_checkpointing:
                try:
                    self.model.gradient_checkpointing_enable()
                except AttributeError:
                    pass  # 모든 모델이 지원하지 않음
    
    def forward(self, *args, **kwargs):
        """최적화된 순전파"""
        model_to_use = self.compiled_model if self.compiled_model else self.model
        
        with global_profiler.profile("forward_pass"):
            return model_to_use(*args, **kwargs)

class AdaptiveBatchSize:
    """적응적 배치 크기 조정"""
    
    def __init__(self, 
                 model: nn.Module,
                 initial_batch_size: int = 64,
                 max_batch_size: int = 512,
                 min_batch_size: int = 8):
        self.model = model
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.oom_count = 0
        self.success_count = 0
        
    def adjust_batch_size(self, oom_occurred: bool = False):
        """배치 크기 조정"""
        if oom_occurred:
            self.oom_count += 1
            self.success_count = 0
            # OOM 발생시 배치 크기 감소
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.75)
            )
            print(f"OOM detected. Reducing batch size to {self.current_batch_size}")
        else:
            self.success_count += 1
            self.oom_count = 0
            # 연속 성공시 배치 크기 증가
            if self.success_count >= 10:
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * 1.1)
                )
                self.success_count = 0
                
    def get_batch_size(self) -> int:
        """현재 배치 크기 반환"""
        return self.current_batch_size

def benchmark_model_performance(model: nn.Module,
                               input_shape: Tuple[int, ...],
                               device: str = "cuda",
                               num_warmup: int = 10,
                               num_iterations: int = 100) -> Dict[str, float]:
    model.eval()
    model = model.to(device)
    
    # 테스트 입력 생성
    dummy_input = torch.randn(input_shape, device=device)
    
    # 워밍업
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # 실제 측정
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # ms
    
    # 메모리 사용량
    if device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        memory_used = 0
    
    # 통계 계산
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = (input_shape[0] * num_iterations) / (sum(times) / 1000)  # samples/sec
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time, 
        'max_time_ms': max_time,
        'throughput_samples_per_sec': throughput,
        'memory_usage_mb': memory_used,
        'total_iterations': num_iterations
    }

def optimize_for_inference(model: nn.Module) -> nn.Module:
    """추론용 모델 최적화
    
    Args:
        model: 최적화할 모델
        
    Returns:
        nn.Module: 최적화된 모델
    """
    # 평가 모드 설정
    model.eval()
    
    # 배치 정규화 레이어 융합 (가능한 경우)
    try:
        model = torch.jit.script(model)
        print("Model converted to TorchScript")
    except Exception as e:
        warnings.warn(f"TorchScript conversion failed: {e}")
    
    return model

class MemoryOptimizer:
    """메모리 사용량 최적화 도구"""
    
    @staticmethod
    def clear_cache():
        """CUDA 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, int]:
        """메모리 사용량 통계"""
        if not torch.cuda.is_available():
            return {"message": "CUDA not available"}
            
        return {
            "allocated_mb": torch.cuda.memory_allocated() // 1024 // 1024,
            "reserved_mb": torch.cuda.memory_reserved() // 1024 // 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() // 1024 // 1024,
            "max_reserved_mb": torch.cuda.max_memory_reserved() // 1024 // 1024
        }
    
    @staticmethod 
    def reset_peak_stats():
        """최대 메모리 사용량 통계 리셋"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

def create_optimized_config_for_task(task: str) -> OptimizationConfig:
    """작업별 최적화 설정 생성
    
    Args:
        task: "training", "inference", "research", "production"
        
    Returns:
        OptimizationConfig: 최적화 설정
    """
    if task == "training":
        return OptimizationConfig(
            use_memory_efficient=True,
            gradient_checkpointing=True,
            mixed_precision=True,
            prefer_fused_ops=True,
            adaptive_batch_size=True,
            use_torch_compile=False  # 훈련시 안정성 우선
        )
    elif task == "inference": 
        return OptimizationConfig(
            use_memory_efficient=True,
            gradient_checkpointing=False,
            mixed_precision=True,
            prefer_fused_ops=True,
            adaptive_batch_size=False,
            use_torch_compile=True  # 추론시 속도 우선
        )
    elif task == "research":
        return OptimizationConfig(
            use_memory_efficient=False,  # 디버깅 편의성
            gradient_checkpointing=False,
            mixed_precision=False,
            prefer_fused_ops=False,  # 개별 연산 분석 가능
            adaptive_batch_size=False,
            use_torch_compile=False
        )
    elif task == "production":
        return OptimizationConfig(
            use_memory_efficient=True,
            gradient_checkpointing=False,
            mixed_precision=True,
            prefer_fused_ops=True,
            adaptive_batch_size=True,
            use_torch_compile=True,
            compile_mode="max-autotune"  # 최고 성능
        )
    else:
        return OptimizationConfig()  # 기본 설정

# ===============================
# Quick Setup Functions  
# ===============================

def quick_setup_for_mnist():
    """MNIST 용 빠른 설정"""
    config = create_optimized_config_for_task("training")
    setup_optimizations(config)
    enable_profiling()
    print("Reality Stone optimized for MNIST training")

def quick_setup_for_production():
    """프로덕션 용 빠른 설정"""
    config = create_optimized_config_for_task("production")
    setup_optimizations(config)
    disable_profiling()  # 프로덕션에서는 프로파일링 비활성화
    print("Reality Stone optimized for production")

def quick_setup_for_research():
    """연구 용 빠른 설정"""
    config = create_optimized_config_for_task("research")
    setup_optimizations(config)
    enable_profiling()
    print("Reality Stone optimized for research") 