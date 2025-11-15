# Models module
# Sentence-Topic LLM modules
try:
    from .sentence_topic_head import SentenceTopicHead
    from .metric_router import MetricContextRouter
    from .rce_lexical_decoder import RCELexicalDecoder
    __all__ = ["SentenceTopicHead", "MetricContextRouter", "RCELexicalDecoder"]
except ImportError:
    __all__ = []

# Legacy modules (optional)
try:
    from .intent_clf import IntentClassifier, RiemannEncoderBlock, MeanMaxPooler, SinusoidalPositionalEncoding
    from .rce_transformer import RCETransformerLM, TopDownRCETransformerLM, count_parameters
    __all__.extend([
        "IntentClassifier", "RiemannEncoderBlock", "MeanMaxPooler", 
        "SinusoidalPositionalEncoding", "RCETransformerLM", 
        "TopDownRCETransformerLM", "count_parameters"
    ])
except ImportError:
    pass
