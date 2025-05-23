import torch
import torch.nn as nn
import torch.nn.functional as F
import reality_stone as rs
from transformers import GPT2Model, GPT2Config, AutoTokenizer
import math
from typing import Optional, Dict, Any

# 이전 코드의 HelgasonFourierTransform 클래스를 import 한다고 가정

# ==================== Reality Stone 기반 LLM 변환 ====================

class RealityStoneGPT2(nn.Module):
    """Reality Stone을 사용하는 하이퍼볼릭 GPT-2"""
    
    def __init__(self, config, c=0.1, manifold='poincare', use_helgason=True):
        super().__init__()
        self.config = config
        self.c = c
        self.manifold = manifold
        self.use_helgason = use_helgason
        
        # 임베딩 레이어
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Reality Stone 기반 트랜스포머 블록들
        self.h = nn.ModuleList([
            RealityStoneTransformerBlock(
                config.n_embd,
                config.n_head,
                config.n_embd * 4,
                c=c,
                manifold=manifold,
                use_helgason=use_helgason,
                dropout=config.resid_pdrop
            )
            for _ in range(config.n_layer)
        ])
        
        # 최종 레이어 정규화
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 헬가손 위치 인코딩
        if use_helgason:
            self.pos_helgason = HelgasonFourierTransform(
                config.n_embd,
                num_frequencies=64,
                c=c,
                manifold=manifold
            )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: bool = False,
    ):
        # 입력 처리
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        
        # 위치 ID
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 임베딩
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # 유클리드에서 하이퍼볼릭으로 변환 (exp_map)
        # Reality Stone의 exp_map은 원점 기준
        hidden_norm = hidden_states.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        hidden_states = torch.tanh(math.sqrt(self.c) * hidden_norm) / (math.sqrt(self.c) * hidden_norm) * hidden_states
        
        # 헬가손 위치 인코딩 추가
        if self.use_helgason and hasattr(self, 'pos_helgason'):
            pos_features = self.pos_helgason(hidden_states)
            # 위치 특징을 부드럽게 통합
            hidden_tangent = self._log_map_0(hidden_states)
            hidden_tangent = hidden_tangent + pos_features[..., :self.config.n_embd] * 0.1
            hidden_states = self._exp_map_0(hidden_tangent)
        
        # 어텐션 마스크 준비
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 트랜스포머 블록들 통과
        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states, attention_mask)
        
        # 최종 정규화 (접공간에서)
        hidden_tangent = self._log_map_0(hidden_states)
        hidden_states = self.ln_f(hidden_tangent)
        
        return hidden_states
    
    def _exp_map_0(self, v):
        """원점에서의 지수 맵"""
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.tanh(math.sqrt(self.c) * v_norm) / (math.sqrt(self.c) * v_norm) * v
    
    def _log_map_0(self, x):
        """원점으로의 로그 맵"""
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.atanh(math.sqrt(self.c) * x_norm.clamp_max(1-1e-7)) / (math.sqrt(self.c) * x_norm) * x


class RealityStoneTransformerBlock(nn.Module):
    """Reality Stone 기반 트랜스포머 블록"""
    
    def __init__(self, d_model, n_heads, d_ff, c=0.1, manifold='poincare', 
                 use_helgason=True, dropout=0.1):
        super().__init__()
        self.c = c
        self.manifold = manifold
        
        # 레이어 정규화
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Reality Stone 하이퍼볼릭 어텐션
        self.attn = RealityStoneAttention(
            d_model, n_heads, c, manifold, use_helgason, dropout
        )
        
        # Reality Stone FFN
        self.mlp = RealityStoneFFN(d_model, d_ff, c, manifold, use_helgason, dropout)
        
    def forward(self, x, attention_mask=None):
        # 어텐션 블록
        # 1. 접공간에서 정규화
        x_tangent = self._log_map_0(x)
        normed = self.ln_1(x_tangent)
        normed_hyp = self._exp_map_0(normed)
        
        # 2. 하이퍼볼릭 어텐션
        attn_output = self.attn(normed_hyp, attention_mask)
        
        # 3. Reality Stone의 Möbius 덧셈으로 잔차 연결
        if x.is_cuda:
            x = rs.mobius_add_cuda(x, attn_output, self.c)
        else:
            x = rs.mobius_add_cpu(x, attn_output, self.c)
        
        # FFN 블록
        # 4. 접공간에서 정규화
        x_tangent = self._log_map_0(x)
        normed = self.ln_2(x_tangent)
        normed_hyp = self._exp_map_0(normed)
        
        # 5. 하이퍼볼릭 FFN
        ffn_output = self.mlp(normed_hyp)
        
        # 6. Möbius 덧셈으로 잔차 연결
        if x.is_cuda:
            x = rs.mobius_add_cuda(x, ffn_output, self.c)
        else:
            x = rs.mobius_add_cpu(x, ffn_output, self.c)
        
        return x
    
    def _exp_map_0(self, v):
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.tanh(math.sqrt(self.c) * v_norm) / (math.sqrt(self.c) * v_norm) * v
    
    def _log_map_0(self, x):
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.atanh(math.sqrt(self.c) * x_norm.clamp_max(1-1e-7)) / (math.sqrt(self.c) * x_norm) * x


class RealityStoneAttention(nn.Module):
    """Reality Stone 연산을 사용하는 효율적인 하이퍼볼릭 어텐션"""
    
    def __init__(self, d_model, n_heads, c=0.1, manifold='poincare', 
                 use_helgason=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.c = c
        self.manifold = manifold
        self.use_helgason = use_helgason
        
        # QKV 프로젝션 (한번에)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        if use_helgason:
            self.helgason = HelgasonFourierTransform(
                self.d_head, num_frequencies=16, c=c, manifold=manifold
            )
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. QKV 계산 (접공간에서)
        hidden_tangent = self._log_map_0(hidden_states)
        qkv = self.c_attn(hidden_tangent)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # 2. 헤드 분리 및 하이퍼볼릭 매핑
        q = self._split_heads(q, batch_size)  # [B, H, S, D]
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)
        
        # 하이퍼볼릭 공간으로 (배치 처리)
        q = self._exp_map_0(q)
        k = self._exp_map_0(k)
        v = self._exp_map_0(v)
        
        # 3. 어텐션 계산
        if self.use_helgason:
            attn_output = self._helgason_attention(q, k, v, attention_mask)
        else:
            attn_output = self._hyperbolic_attention(q, k, v, attention_mask)
        
        # 4. 헤드 병합
        attn_output = self._merge_heads(attn_output, batch_size)
        
        # 5. 출력 프로젝션 (접공간에서)
        attn_tangent = self._log_map_0(attn_output)
        attn_output = self.c_proj(attn_tangent)
        attn_output = self.resid_dropout(attn_output)
        
        # 6. 하이퍼볼릭 공간으로
        attn_output = self._exp_map_0(attn_output)
        
        return attn_output
    
    def _helgason_attention(self, q, k, v, mask):
        """헬가손 변환을 사용한 효율적인 어텐션"""
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # 헬가손 변환 (각 헤드별로)
        q_flat = q.transpose(1, 2).reshape(-1, seq_len, d_head)
        k_flat = k.transpose(1, 2).reshape(-1, seq_len, d_head)
        
        q_freq = self.helgason(q_flat)
        k_freq = self.helgason(k_flat)
        
        # 주파수 영역에서 내적
        scores = torch.matmul(q_freq, k_freq.transpose(-2, -1))
        scores = scores.view(batch_size, n_heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_head)
        
        # 마스크 적용
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 값 집계 (Einstein 중점)
        attn_output = self._einstein_midpoint(attn_weights, v)
        
        return attn_output
    
    def _hyperbolic_attention(self, q, k, v, mask):
        """표준 하이퍼볼릭 어텐션"""
        # 간단한 구현: 접공간에서 내적
        q_tangent = self._log_map_0(q)
        k_tangent = self._log_map_0(k)
        
        scores = torch.matmul(q_tangent, k_tangent.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Einstein 중점으로 집계
        attn_output = self._einstein_midpoint(attn_weights, v)
        
        return attn_output
    
    def _einstein_midpoint(self, weights, values):
        """Einstein 중점을 사용한 가중 평균"""
        # 접공간에서 가중 평균
        v_tangent = self._log_map_0(values)
        weighted_sum = torch.matmul(weights, v_tangent)
        
        # 하이퍼볼릭 공간으로
        return self._exp_map_0(weighted_sum)
    
    def _split_heads(self, x, batch_size):
        """헤드 분리"""
        x = x.view(batch_size, -1, self.n_heads, self.d_head)
        return x.transpose(1, 2)  # [B, H, S, D]
    
    def _merge_heads(self, x, batch_size):
        """헤드 병합"""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def _exp_map_0(self, v):
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.tanh(math.sqrt(self.c) * v_norm) / (math.sqrt(self.c) * v_norm) * v
    
    def _log_map_0(self, x):
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.atanh(math.sqrt(self.c) * x_norm.clamp_max(1-1e-7)) / (math.sqrt(self.c) * x_norm) * x


class RealityStoneFFN(nn.Module):
    """Reality Stone 기반 Feed-Forward Network"""
    
    def __init__(self, d_model, d_ff, c=0.1, manifold='poincare', 
                 use_helgason=True, dropout=0.1):
        super().__init__()
        self.c = c
        self.manifold = manifold
        self.use_helgason = use_helgason
        
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_helgason:
            self.helgason_act = HelgasonFourierTransform(
                d_ff, num_frequencies=32, c=c, manifold=manifold
            )
        
    def forward(self, hidden_states):
        # 1. 접공간으로
        hidden_tangent = self._log_map_0(hidden_states)
        
        # 2. 첫 번째 선형 변환
        h = self.c_fc(hidden_tangent)
        
        # 3. 하이퍼볼릭 공간으로
        h = self._exp_map_0(h)
        
        # 4. 활성화 함수
        if self.use_helgason:
            # 헬가손 변환으로 비선형성
            h_freq = self.helgason_act(h)
            # 주파수 영역에서 GELU
            h_freq = F.gelu(h_freq)
            # 역변환
            h = self.helgason_act.inverse_transform(h_freq)
        else:
            # Reality Stone의 측지선 보간을 활용한 활성화
            # GELU 근사: h와 활성화된 h 사이의 측지선 보간
            h_tangent = self._log_map_0(h)
            h_act = F.gelu(h_tangent)
            h_act_hyp = self._exp_map_0(h_act)
            
            # 측지선 보간 (t=0.7)
            if h.is_cuda:
                h = rs.poincare_ball_forward_cuda(h, h_act_hyp, self.c, 0.7)
            else:
                h = rs.poincare_ball_forward_cpu(h, h_act_hyp, self.c, 0.7)
        
        # 5. 접공간으로
        h_tangent = self._log_map_0(h)
        
        # 6. 두 번째 선형 변환
        h = self.c_proj(h_tangent)
        h = self.dropout(h)
        
        # 7. 하이퍼볼릭 공간으로
        return self._exp_map_0(h)
    
    def _exp_map_0(self, v):
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.tanh(math.sqrt(self.c) * v_norm) / (math.sqrt(self.c) * v_norm) * v
    
    def _log_map_0(self, x):
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.atanh(math.sqrt(self.c) * x_norm.clamp_max(1-1e-7)) / (math.sqrt(self.c) * x_norm) * x


# ==================== GPT-2 변환 유틸리티 ====================

def convert_gpt2_to_hyperbolic(pretrained_model_name='gpt2', c=0.1, use_helgason=True):
    """사전학습된 GPT-2를 Reality Stone 하이퍼볼릭 버전으로 변환"""
    
    print(f"Loading pretrained model: {pretrained_model_name}")
    
    # 원본 모델과 설정 로드
    from transformers import GPT2Model
    original_model = GPT2Model.from_pretrained(pretrained_model_name)
    config = original_model.config
    
    # Reality Stone GPT-2 생성
    print(f"Creating hyperbolic model with c={c}, use_helgason={use_helgason}")
    hyperbolic_model = RealityStoneGPT2(config, c=c, use_helgason=use_helgason)
    
    # 가중치 복사
    print("Copying weights...")
    
    # 임베딩 가중치
    hyperbolic_model.wte.weight.data = original_model.wte.weight.data.clone()
    hyperbolic_model.wpe.weight.data = original_model.wpe.weight.data.clone()
    
    # 각 트랜스포머 블록
    for i, (orig_block, hyp_block) in enumerate(zip(original_model.h, hyperbolic_model.h)):
        print(f"Converting block {i+1}/{len(original_model.h)}")
        
        # LayerNorm
        hyp_block.ln_1.weight.data = orig_block.ln_1.weight.data.clone()
        hyp_block.ln_1.bias.data = orig_block.ln_1.bias.data.clone()
        hyp_block.ln_2.weight.data = orig_block.ln_2.weight.data.clone()
        hyp_block.ln_2.bias.data = orig_block.ln_2.bias.data.clone()
        
        # Attention
        hyp_block.attn.c_attn.weight.data = orig_block.attn.c_attn.weight.data.clone()
        hyp_block.attn.c_attn.bias.data = orig_block.attn.c_attn.bias.data.clone()
        hyp_block.attn.c_proj.weight.data = orig_block.attn.c_proj.weight.data.clone()
        hyp_block.attn.c_proj.bias.data = orig_block.attn.c_proj.bias.data.clone()
        
        # MLP
        hyp_block.mlp.c_fc.weight.data = orig_block.mlp.c_fc.weight.data.clone()
        hyp_block.mlp.c_fc.bias.data = orig_block.mlp.c_fc.bias.data.clone()
        hyp_block.mlp.c_proj.weight.data = orig_block.mlp.c_proj.weight.data.clone()
        hyp_block.mlp.c_proj.bias.data = orig_block.mlp.c_proj.bias.data.clone()
    
    # 최종 LayerNorm
    hyperbolic_model.ln_f.weight.data = original_model.ln_f.weight.data.clone()
    hyperbolic_model.ln_f.bias.data = original_model.ln_f.bias.data.clone()
    
    print("Weight copying complete!")
    
    return hyperbolic_model


# ==================== 텍스트 생성 ====================

class HyperbolicTextGenerator:
    """Reality Stone 하이퍼볼릭 모델을 위한 텍스트 생성기"""
    
    def __init__(self, model, tokenizer, c=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.c = c
        
        # LM 헤드 추가
        self.lm_head = nn.Linear(model.config.n_embd, model.config.vocab_size, bias=False)
        
        # GPT-2의 wte 가중치와 공유 (weight tying)
        self.lm_head.weight = model.wte.weight
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50):
        """텍스트 생성"""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 프롬프트 토큰화
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # 모델 실행
                outputs = self.model(input_ids)
                
                # 마지막 토큰의 hidden state
                last_hidden = outputs[:, -1, :]
                
                # 접공간으로 변환하여 로짓 계산
                last_hidden_tangent = self._log_map_0(last_hidden)
                logits = self.lm_head(last_hidden_tangent)
                
                # Temperature 적용
                logits = logits / temperature
                
                # Top-k 샘플링
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # 확률 계산 및 샘플링
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 토큰 추가
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # EOS 토큰 확인
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 디코딩
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    def _log_map_0(self, x):
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        return torch.atanh(math.sqrt(self.c) * x_norm.clamp_max(1-1e-7)) / (math.sqrt(self.c) * x_norm) * x


# ==================== 실제 사용 예시 ====================

if __name__ == "__main__":
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. GPT-2 변환
    print("\n=== Converting GPT-2 to Hyperbolic ===")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 작은 모델로 테스트 (메모리 절약)
    # 실제로는 'gpt2', 'gpt2-medium', 'gpt2-large' 등 사용 가능
    hyperbolic_gpt2 = convert_gpt2_to_hyperbolic(
        'gpt2',  # 또는 'gpt2-medium', 'gpt2-large'
        c=0.1,
        use_helgason=True
    ).to(device)
    
    # 2. 모델 정보
    print("\n=== Model Information ===")
    total_params = sum(p.numel() for p in hyperbolic_gpt2.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Memory usage: {total_params * 4 / (1024**3):.2f} GB (FP32)")
    
    # 3. 간단한 추론 테스트
    print("\n=== Inference Test ===")
    test_text = "The hyperbolic geometry"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = hyperbolic_gpt2(inputs['input_ids'])
        print(f"Output shape: {outputs.shape}")
        print(f"Output norm (should be < 1/sqrt(c)): {outputs.norm(dim=-1).mean().item():.4f}")
        print(f"Max allowed norm: {1/math.sqrt(0.1):.4f}")
    
    # 4. 텍스트 생성
    print("\n=== Text Generation ===")
    generator = HyperbolicTextGenerator(hyperbolic_gpt2, tokenizer, c=0.1)
    
    prompts = [
        "The future of AI is",
        "Hyperbolic neural networks can",
        "In the realm of geometry,",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generator.generate(prompt, max_length=30, temperature=0.8)
        print(f"Generated: {generated}")
    
    # 5. Reality Stone 연산 벤치마크
    print("\n=== Reality Stone Operations Benchmark ===")
    
    # 테스트 데이터
    batch_size = 32
    seq_len = 128
    d_model = 768
    
    x = torch.randn(batch_size, seq_len, d_model).to(device) * 0.1
    y = torch.randn(batch_size, seq_len, d_model).to(device) * 0.1
    
    # Möbius 덧셈 시간 측정
    import time
    
    # Warm up
    for _ in range(10):
        if device.type == 'cuda':
            _ = rs.mobius_add_cuda(x[0:1], y[0:1], 0.1)
        else:
            _ = rs.mobius_add_cpu(x[0:1], y[0:1], 0.1)
    
    # 실제 측정
    start = time.time()
    for _ in range(100):
        if device.type == 'cuda':
            result = rs.mobius_add_cuda(x, y, 0.1)
        else:
            result = rs.mobius_add_cpu(x, y, 0.1)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    end = time.time()
    print(f"Möbius addition (100 iterations): {(end-start)*1000:.2f} ms")
    print(f"Per operation: {(end-start)*10:.2f} ms")
    
    # 메모리 사용량
    if device.type == 'cuda':
        print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(device)/(1024**2):.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_cached(device)/(1024**2):.2f} MB")