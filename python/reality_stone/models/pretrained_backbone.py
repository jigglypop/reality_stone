import torch
import torch.nn as nn
from typing import Optional

class PretrainedBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "klue/bert-base",
        freeze: bool = True,
        d_model: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.d_model = d_model
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.backbone = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            backbone_dim = self.backbone.config.hidden_size
            if backbone_dim != d_model:
                self.proj = nn.Linear(backbone_dim, d_model)
            else:
                self.proj = nn.Identity()
            
            if freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"[PretrainedBackbone] Loaded {model_name}, frozen")
            else:
                print(f"[PretrainedBackbone] Loaded {model_name}, trainable")
        except Exception as e:
            print(f"[PretrainedBackbone] Failed: {e}")
            print("[PretrainedBackbone] Random init")
            self.backbone = None
            self.tokenizer = None
            self.proj = nn.Identity()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            B, S = input_ids.shape
            return torch.randn(B, S, self.d_model, device=input_ids.device)
        outputs = self.backbone(input_ids, return_dict=True)
        hidden = outputs.last_hidden_state
        embeddings = self.proj(hidden)
        return embeddings
    
    def get_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer)
        return 32000

