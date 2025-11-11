import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.activations import ACT2FN
from typing import Optional, Tuple

from ..layers.lowrank import LowRankLinear


class LowRankConv1D(nn.Module):
    """1D-convolutional layer as used by GPT2 with low-rank decomposition."""
    
    def __init__(self, nf, nx, r=64):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.r = r
        
        # Low-rank decomposition: W = Q @ Sigma @ P^T
        self.P = nn.Parameter(torch.empty(nx, r))
        self.Q = nn.Parameter(torch.empty(nf, r))
        self.Sigma = nn.Parameter(torch.empty(r, r))
        self.bias = nn.Parameter(torch.zeros(nf))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.P)
        nn.init.orthogonal_(self.Q)
        nn.init.eye_(self.Sigma)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: [batch, seq_len, nx]
        # Compute: y = x @ P @ Sigma^T @ Q^T + bias
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, self.nx)
        
        # Low-rank matmul
        x = torch.matmul(x, self.P)         # [batch*seq, r]
        x = torch.matmul(x, self.Sigma.t()) # [batch*seq, r]
        x = torch.matmul(x, self.Q.t())     # [batch*seq, nf]
        x = x + self.bias
        
        x = x.view(size_out)
        return x
    
    @classmethod
    def from_conv1d(cls, conv1d, r=64):
        """Convert standard Conv1D to LowRankConv1D"""
        nf = conv1d.weight.shape[1]
        nx = conv1d.weight.shape[0]
        
        layer = cls(nf, nx, r=r)
        
        # Perform SVD on weight matrix
        W = conv1d.weight.detach().t()  # [nf, nx]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        r_actual = min(r, U.shape[1], Vh.shape[0])
        U_r = U[:, :r_actual]
        V_r = Vh[:r_actual, :].t()
        S_r = torch.diag(S[:r_actual])
        
        # Pad if necessary
        if r_actual < r:
            pad_size = r - r_actual
            U_r = torch.cat([U_r, torch.zeros(nf, pad_size)], dim=1)
            V_r = torch.cat([V_r, torch.zeros(nx, pad_size)], dim=1)
            S_r = torch.nn.functional.pad(S_r, (0, pad_size, 0, pad_size))
        
        with torch.no_grad():
            layer.P.copy_(V_r)
            layer.Q.copy_(U_r)
            layer.Sigma.copy_(S_r)
            layer.bias.copy_(conv1d.bias)
        
        return layer


class LowRankGPT2Attention(nn.Module):
    """GPT2 attention with low-rank linear layers"""
    
    def __init__(self, config, is_cross_attention=False, layer_idx=None, rank=64):
        super().__init__()
        self.config = config
        self.rank = rank
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Low-rank layers
        if self.is_cross_attention:
            self.c_attn = LowRankConv1D(2 * self.embed_dim, self.embed_dim, r=rank)
            self.q_attn = LowRankConv1D(self.embed_dim, self.embed_dim, r=rank)
        else:
            self.c_attn = LowRankConv1D(3 * self.embed_dim, self.embed_dim, r=rank)
        self.c_proj = LowRankConv1D(self.embed_dim, self.embed_dim, r=rank)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,  # For compatibility
    ):
        # Standard GPT2 attention forward, just with low-rank layers
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class LowRankGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, rank=64):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = LowRankConv1D(intermediate_size, embed_dim, r=rank)
        self.c_proj = LowRankConv1D(embed_dim, intermediate_size, r=rank)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def convert_gpt2_to_lowrank(model, rank=64):
    """Convert a GPT2 model to use low-rank layers"""
    config = model.config
    
    # Convert each transformer block
    for i, block in enumerate(model.transformer.h):
        # Convert attention
        old_attn = block.attn
        new_attn = LowRankGPT2Attention(config, layer_idx=i, rank=rank)
        
        # Copy weights with SVD
        new_attn.c_attn = LowRankConv1D.from_conv1d(old_attn.c_attn, r=rank)
        new_attn.c_proj = LowRankConv1D.from_conv1d(old_attn.c_proj, r=rank)
        
        block.attn = new_attn
        
        # Convert MLP
        old_mlp = block.mlp
        intermediate_size = old_mlp.c_fc.weight.shape[1]
        new_mlp = LowRankGPT2MLP(intermediate_size, config, rank=rank)
        
        new_mlp.c_fc = LowRankConv1D.from_conv1d(old_mlp.c_fc, r=rank)
        new_mlp.c_proj = LowRankConv1D.from_conv1d(old_mlp.c_proj, r=rank)
        
        block.mlp = new_mlp
    
    # Don't convert lm_head - keep it full rank
    # model.lm_head remains unchanged
    
    return model
