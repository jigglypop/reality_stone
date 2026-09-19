"""Model namespace with lazy imports.

``hierarchical_sentence_topic_llm`` (2,300+ lines, pulls in ``transformers``) and
``transformer_converter`` are only imported when one of their names is first
accessed, so ``import reality_stone`` stays cheap. Names that fail to import
resolve to ``None`` exactly as before; the import error is kept in
``_IMPORT_ERRORS`` for diagnosis.
"""

from __future__ import annotations

import importlib
from typing import Any

from .residual_reuse import (
    ResidualReuseMLP,
    ResidualReuseStats,
    install_gpt2_residual_reuse,
    reset_residual_reuse,
    residual_reuse_report,
)
from .riemannian_aggregation import RiemannianAggregation

_LAZY: dict[str, str] = {
    "HierarchicalLLMConfig": ".hierarchical_sentence_topic_llm",
    "HierarchicalSentenceTopicLLM": ".hierarchical_sentence_topic_llm",
    "SentenceTopicHead": ".hierarchical_sentence_topic_llm",
    "MetricContextRouter": ".hierarchical_sentence_topic_llm",
    "HierarchicalLMDecoder": ".hierarchical_sentence_topic_llm",
    "RCELexicalDecoder": ".hierarchical_sentence_topic_llm",
    "HAS_METRIKEY": ".hierarchical_sentence_topic_llm",
    "RSULFConfig": ".transformer_converter",
    "RSULFTransformerConverter": ".transformer_converter",
    "convert_transformer_to_rsulf": ".transformer_converter",
}
_FLAGS = {"_HAS_LLM": ".hierarchical_sentence_topic_llm", "_HAS_CONVERTER": ".transformer_converter"}
_IMPORT_ERRORS: dict[str, Exception] = {}


def _load(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name, __name__)
    except ImportError as error:
        _IMPORT_ERRORS[module_name] = error
        return None


def __getattr__(name: str) -> Any:
    if name in _FLAGS:
        return _load(_FLAGS[name]) is not None
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _load(module_name)
    if module is None:
        return False if name == "HAS_METRIKEY" else None
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_FLAGS))


__all__ = [
    "RiemannianAggregation",
    "ResidualReuseMLP",
    "ResidualReuseStats",
    "install_gpt2_residual_reuse",
    "reset_residual_reuse",
    "residual_reuse_report",
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
    "_HAS_LLM",
    "_HAS_CONVERTER",
]
