# Import core classes directly from advanced.py
from ..advanced import (
    AdvancedConfig,
    predict_dynamic_curvature,
    dynamic_mobius_add,
    hyperbolic_regularization,
    geodesic_activation,
    einstein_midpoint,
    hyperbolic_linear_fused,
    transform_regularize_fused,
    fix_mnist_nan,
    chebyshev_approximation,
    hyperbolic_fft,
    create_advanced_config,
    get_available_features,
    benchmark_advanced_features
)

# Future modules
from .spectral import *
from .compression import *

__all__ = [
    'AdvancedConfig',
    'predict_dynamic_curvature',
    'dynamic_mobius_add',
    'hyperbolic_regularization',
    'geodesic_activation',
    'einstein_midpoint',
    'hyperbolic_linear_fused',
    'transform_regularize_fused',
    'fix_mnist_nan',
    'chebyshev_approximation',
    'hyperbolic_fft',
    'create_advanced_config',
    'get_available_features',
    'benchmark_advanced_features'
] 