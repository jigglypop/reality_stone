import pytest
import torch

from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    infer_hierarchical_llm_on_text,
)


@pytest.fixture
def sample_config():
    """기본 테스트용 HierarchicalLLMConfig 생성"""
    return HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=8,
        num_heads_topic=2,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )


@pytest.fixture
def sample_model(sample_config):
    """기본 테스트용 HierarchicalSentenceTopicLLM 모델 생성"""
    return HierarchicalSentenceTopicLLM(sample_config)


def test_recursive_segment():
    """
    PreSegmenter가 계층적 문서 구조를 올바르게 세그먼트하는지 검증.
    문서 → 섹션 → 서브섹션 → 문장 → 토큰 레벨 분리 확인.
    """
    segmenter = PreSegmenter()
    text = "Section1\n\nSubsection1.1\nSentence one. Sentence two.\n\nSubsection1.2\nAnother sentence."
    levels = ['document', 'section', 'subsection', 'sentence', 'token']
    tree = segmenter.recursive_segment(text, levels)
    
    assert len(tree.nodes) > 5, "트리 노드가 최소 6개 이상이어야 함"
    assert tree.nodes[0].type == 'document', "루트 노드는 document 타입이어야 함"
    
    sections = tree.children(0)
    assert len(sections) == 1, "1개의 섹션이 있어야 함"
    
    subsections = tree.children(sections[0])
    assert len(subsections) == 2, "2개의 서브섹션이 있어야 함"


def test_full_edit_ops(sample_model):
    """
    구조적 편집 연산이 활성화된 경우 추론이 정상 동작하는지 검증.
    enable_structural_edit=True 시 생성된 텍스트 확인.
    """
    sample_model.config.enable_structural_edit = True
    sample_model.eval()
    
    with torch.no_grad():
        out = infer_hierarchical_llm_on_text(
            sample_model, 
            "Test text with three sentences.", 
            max_length=5
        )
    
    generated = out['generated_text']
    assert isinstance(generated, str), "생성된 텍스트는 문자열이어야 함"
    assert len(generated.strip()) > 0, "생성된 텍스트는 비어있지 않아야 함"


def test_pretrain_loading():
    """
    freeze_decoder=True 옵션이 디코더 파라미터를 올바르게 동결하는지 검증.
    """
    config = HierarchicalLLMConfig(
        freeze_decoder=True,
        use_pretrained_embeddings=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    frozen_params = [p for p in model.decoder.parameters() if not p.requires_grad]
    all_decoder_params = list(model.decoder.parameters())
    
    assert len(frozen_params) == len(all_decoder_params), \
        "모든 디코더 파라미터가 동결되어야 함"


def test_model_initialization(sample_config):
    """
    모델이 주어진 config로 올바르게 초기화되는지 검증.
    """
    model = HierarchicalSentenceTopicLLM(sample_config)
    
    assert model.config.vocab_size == sample_config.vocab_size
    assert model.config.d_model == sample_config.d_model
    assert model.config.d_head == sample_config.d_head
    
    assert hasattr(model, 'sentence_aggregator')
    assert hasattr(model, 'paragraph_aggregator')
    assert hasattr(model, 'topic_head')
    assert hasattr(model, 'metric_router')
    assert hasattr(model, 'decoder')


def test_forward_pass_shape(sample_model):
    """
    모델의 forward pass가 올바른 shape의 출력을 생성하는지 검증.
    """
    B, T, L = 2, 3, 10
    tokens = torch.randint(1, sample_model.config.vocab_size, (B, T, L))
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    batch = {"tokens": tokens, "topo_idx": topo_idx}
    
    sample_model.eval()
    with torch.no_grad():
        logits, info = sample_model(batch, compute_loss=False)
    
    assert "P_topic" in info
    assert "scores" in info
    assert "metric_ctx" in info
    
    P_topic = info["P_topic"]
    assert P_topic.shape[0] == B
    assert P_topic.shape[1] == T
    assert P_topic.shape[2] == sample_model.config.num_topics


def test_encode_tokens_to_sentences(sample_model):
    """
    토큰 → 문장 인코딩이 올바른 shape을 반환하는지 검증.
    """
    B, T, L = 2, 3, 5
    tokens = torch.randint(1, sample_model.config.vocab_size, (B, T, L))
    
    sample_model.eval()
    with torch.no_grad():
        sentence_embeddings = sample_model.encode_tokens_to_sentences(tokens)
    
    assert sentence_embeddings.shape == (B, T, sample_model.config.d_model)


def test_encode_sentences_to_paragraph(sample_model):
    """
    문장 → 문단 인코딩이 올바른 shape을 반환하는지 검증.
    """
    B, T = 2, 3
    sentence_embeddings = torch.randn(B, T, sample_model.config.d_model)
    
    sample_model.eval()
    with torch.no_grad():
        paragraph_embedding = sample_model.encode_sentences_to_paragraph(
            sentence_embeddings
        )
    
    assert paragraph_embedding.shape == (B, sample_model.config.d_model)

