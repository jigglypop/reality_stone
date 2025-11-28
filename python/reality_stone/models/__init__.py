try:
    from .hierarchical_sentence_topic_llm import (
        HierarchicalLLMConfig,
        HierarchicalSentenceTopicLLM,
        SentenceTopicHead,
        MetricContextRouter,
        HierarchicalLMDecoder,
        RCELexicalDecoder,
        HAS_METRIKEY,
    )
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    HierarchicalLLMConfig = None
    HierarchicalSentenceTopicLLM = None
    SentenceTopicHead = None
    MetricContextRouter = None
    HierarchicalLMDecoder = None
    RCELexicalDecoder = None
    HAS_METRIKEY = False

try:
    from .transformer_converter import (
        RSULFConfig,
        RSULFTransformerConverter,
        convert_transformer_to_rsulf,
    )
    _HAS_CONVERTER = True
except ImportError:
    _HAS_CONVERTER = False
    RSULFConfig = None
    RSULFTransformerConverter = None
    convert_transformer_to_rsulf = None

try:
    from .llm_adapter import (
        LLMAdapterConfig,
        RSULFLLMAdapter,
    )
    _HAS_ADAPTER = True
except ImportError:
    _HAS_ADAPTER = False
    LLMAdapterConfig = None
    RSULFLLMAdapter = None

from .spd_operations import (
    spd_barycenter,
    spd_log,
    spd_exp,
    spd_distance,
    SPDBarycentricMixer,
    CrossLevelMetricMixer,
)

from .riemannian_aggregation import (
    RiemannianAggregation,
)

from .product_manifold import (
    ProductManifold,
    ProductManifoldLayer,
)

__all__ = [
    "spd_barycenter",
    "spd_log",
    "spd_exp",
    "spd_distance",
    "SPDBarycentricMixer",
    "CrossLevelMetricMixer",
    "RiemannianAggregation",
    "ProductManifold",
    "ProductManifoldLayer",
    "HierarchicalLLMConfig",
    "HierarchicalSentenceTopicLLM",
    "SentenceTopicHead",
    "MetricContextRouter",
    "HierarchicalLMDecoder",
    "RCELexicalDecoder",
    "HAS_METRIKEY",
    "RSULFConfig",
    "RSULFTransformerConverter",
    "convert_transformer_to_rsulf",
    "LLMAdapterConfig",
    "RSULFLLMAdapter",
    "_HAS_LLM",
    "_HAS_CONVERTER",
    "_HAS_ADAPTER",
]
