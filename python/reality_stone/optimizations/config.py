from dataclasses import dataclass
import warnings

@dataclass
class OptimizationConfig:
    """
    A data class to hold various optimization settings for the model.
    """
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

def create_optimized_config_for_task(task: str) -> OptimizationConfig:
    """
    Creates a pre-defined OptimizationConfig based on the specified task.

    Args:
        task (str): The target task, one of "training", "inference", 
                    "research", or "production".

    Returns:
        OptimizationConfig: A configuration object tailored for the task.
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
        warnings.warn(f"Unknown task '{task}'. Returning default config.")
        return OptimizationConfig() 