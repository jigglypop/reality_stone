from typing import Optional, Union, Dict, Any
from pathlib import Path
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
)


class HierarchicalLLM:
    
    def __init__(
        self,
        model: HierarchicalSentenceTopicLLM,
        config: HierarchicalLLMConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[HierarchicalLLMConfig] = None,
        device: Optional[str] = None,
    ) -> "HierarchicalLLM":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if config is None:
            if "config" in checkpoint:
                config_dict = checkpoint["config"]
                config = HierarchicalLLMConfig(**config_dict)
            else:
                raise ValueError("Config not found in checkpoint")
        
        model = HierarchicalSentenceTopicLLM(config)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, config, device)
    
    @classmethod
    def from_config(
        cls,
        config: Union[HierarchicalLLMConfig, Dict[str, Any]],
        device: Optional[str] = None,
    ) -> "HierarchicalLLM":
        if isinstance(config, dict):
            config = HierarchicalLLMConfig(**config)
        
        model = HierarchicalSentenceTopicLLM(config)
        return cls(model, config, device)
    
    def generate(
        self,
        text: str,
        max_length: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        enable_structural_edit: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        from reality_stone.models.hierarchical_sentence_topic_llm import infer_hierarchical_llm_on_text
        
        self.model.config.enable_structural_edit = enable_structural_edit
        
        params = {
            "model": self.model,
            "text": text,
            "max_length": max_length or 128,
            "k_neighbors": k_neighbors or 3,
            "max_new_tokens": max_new_tokens or 20,
        }
        params.update(kwargs)
        
        with torch.no_grad():
            result = infer_hierarchical_llm_on_text(**params)
        
        return result if return_dict else result["generated_text"]
    
    def __call__(self, text: str, **kwargs):
        return self.generate(text, **kwargs)
    
    def save_pretrained(self, save_path: Union[str, Path]):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "d_head": self.config.d_head,
                "num_topics": self.config.num_topics,
                "num_heads_topic": self.config.num_heads_topic,
                "n_layer_decoder": self.config.n_layer_decoder,
                "n_head_decoder": self.config.n_head_decoder,
                "c_poincare": self.config.c_poincare,
                "c_lorentz": self.config.c_lorentz,
            }
        }
        
        torch.save(checkpoint, save_path / "model.pt")


def pipeline(
    task: str = "text-generation",
    model: Optional[Union[str, Path, HierarchicalLLM]] = None,
    config: Optional[Union[HierarchicalLLMConfig, Dict]] = None,
    device: Optional[str] = None,
    **kwargs
):
    if isinstance(model, HierarchicalLLM):
        llm = model
    elif isinstance(model, (str, Path)):
        llm = HierarchicalLLM.from_pretrained(model, config, device)
    elif config is not None:
        llm = HierarchicalLLM.from_config(config, device)
    else:
        raise ValueError("Either model or config must be provided")

    # task dispatcher
    from .inference import TextGenerator, TextEditor
    from .qa import QuestionAnswerer
    from .indexing import DocumentIndexer

    task_map = {
        "text-generation": TextGenerator,
        "text-editing": TextEditor,
        "question-answering": QuestionAnswerer,
        "document-indexing": DocumentIndexer,
    }
    cls = task_map.get(task)
    if cls is None:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Available: {', '.join(task_map.keys())}"
        )
    return cls(llm, **kwargs)
