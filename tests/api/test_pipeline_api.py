import pytest
import torch
import tempfile
from pathlib import Path

from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import HierarchicalLLMConfig


@pytest.fixture
def small_config():
    return HierarchicalLLMConfig(
        vocab_size=500,
        d_model=64,
        d_head=16,
        num_topics=4,
        num_heads_topic=2,
        n_layer_decoder=1,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )


@pytest.fixture
def sample_model(small_config):
    return HierarchicalLLM.from_config(small_config)


class TestHierarchicalLLM:
    
    def test_from_config(self, small_config):
        model = HierarchicalLLM.from_config(small_config)
        
        assert model.model is not None
        assert model.config == small_config
    
    def test_from_config_dict(self):
        config_dict = {
            "vocab_size": 500,
            "d_model": 64,
            "d_head": 16,
            "num_topics": 4,
            "use_pretrained_embeddings": False,
        }
        
        model = HierarchicalLLM.from_config(config_dict)
        
        assert model.model is not None
        assert model.config.vocab_size == 500
    
    def test_call_inference(self, sample_model):
        text = "테스트 문장입니다."
        
        result = sample_model(text, max_length=32, k_neighbors=2)
        
        assert "original_text" in result
        assert "generated_text" in result
        assert "sentences" in result
        assert "topics" in result
    
    def test_save_and_load(self, sample_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            sample_model.save_pretrained(save_path)
            
            assert (save_path / "model.pt").exists()
            
            loaded = HierarchicalLLM.from_pretrained(save_path / "model.pt")
            
            assert loaded.config.vocab_size == sample_model.config.vocab_size


class TestPipeline:
    
    def test_pipeline_text_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        text = "테스트 문장입니다."
        output = generator(text)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_pipeline_text_editing(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 문장입니다."
        result = editor(text, enable_structural_edit=False)
        
        assert "original" in result
        assert "edited" in result
        assert "topics" in result
    
    def test_pipeline_with_model_instance(self, sample_model):
        generator = pipeline("text-generation", model=sample_model)
        
        text = "테스트 문장입니다."
        output = generator(text)
        
        assert isinstance(output, str)
    
    def test_pipeline_invalid_task(self, small_config):
        with pytest.raises(ValueError):
            pipeline("invalid-task", config=small_config)
    
    def test_pipeline_no_model_or_config(self):
        with pytest.raises(ValueError):
            pipeline("text-generation")


class TestTextGenerator:
    
    def test_single_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        text = "단일 생성 테스트"
        output = generator(text, max_new_tokens=10)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_batch_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        texts = ["첫 번째 문장", "두 번째 문장", "세 번째 문장"]
        outputs = generator.generate_batch(texts, max_new_tokens=10)
        
        assert len(outputs) == 3
        assert all(isinstance(o, str) for o in outputs)


class TestTextEditor:
    
    def test_edit_with_structural_edit(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 텍스트입니다."
        result = editor(text, enable_structural_edit=True)
        
        assert result["original"] == text
        assert isinstance(result["edited"], str)
        assert isinstance(result["topics"], list)
    
    def test_edit_without_structural_edit(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 텍스트입니다."
        result = editor(text, enable_structural_edit=False)
        
        assert result["original"] == text
        assert isinstance(result["edited"], str)
    
    def test_batch_editing(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        texts = ["첫 번째", "두 번째"]
        results = editor.edit_batch(texts)
        
        assert len(results) == 2
        assert all("original" in r and "edited" in r for r in results)

