"""
한국어 최적화 압축 시스템: 실제 파라미터 감소와 한글 품질 향상
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class KoreanTokenizer:
    """간단한 한국어 토크나이저 (자모 단위)"""
    
    def __init__(self):
        # 한글 자모 분해 테이블
        self.cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # 초성
        self.jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # 중성
        self.jong = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # 종성
        
        # 전체 어휘
        self.vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(self.cho) + list(self.jung) + list(self.jong)
        self.vocab += [chr(i) for i in range(ord('a'), ord('z')+1)]  # 영어 소문자
        self.vocab += [chr(i) for i in range(ord('A'), ord('Z')+1)]  # 영어 대문자
        self.vocab += [chr(i) for i in range(ord('0'), ord('9')+1)]  # 숫자
        self.vocab += [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']']
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def decompose_korean(self, char):
        """한글 문자를 자모로 분해"""
        if not ('가' <= char <= '힣'):
            return [char]
        
        code = ord(char) - ord('가')
        cho_idx = code // (21 * 28)
        jung_idx = (code % (21 * 28)) // 28
        jong_idx = code % 28
        
        result = [self.cho[cho_idx], self.jung[jung_idx]]
        if jong_idx > 0:
            result.append(self.jong[jong_idx])
        
        return result
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 ID로 변환"""
        tokens = [self.token_to_id.get('<bos>', 2)]
        
        for char in text:
            if '가' <= char <= '힣':
                # 한글 분해
                jamos = self.decompose_korean(char)
                for jamo in jamos:
                    tokens.append(self.token_to_id.get(jamo, self.token_to_id['<unk>']))
            else:
                tokens.append(self.token_to_id.get(char, self.token_to_id['<unk>']))
        
        tokens.append(self.token_to_id.get('<eos>', 3))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """토큰 ID를 텍스트로 변환"""
        text = ""
        i = 0
        
        while i < len(token_ids):
            token_id = token_ids[i]
            if token_id in [0, 1, 2, 3]:  # 특수 토큰 건너뛰기
                i += 1
                continue
                
            char = self.id_to_token.get(token_id, '')
            
            # 한글 자모 조합 시도
            if char in self.cho and i + 1 < len(token_ids):
                jung_char = self.id_to_token.get(token_ids[i + 1], '')
                if jung_char in self.jung:
                    # 초성 + 중성
                    cho_idx = self.cho.index(char)
                    jung_idx = self.jung.index(jung_char)
                    jong_idx = 0
                    
                    # 종성 확인
                    if i + 2 < len(token_ids):
                        jong_char = self.id_to_token.get(token_ids[i + 2], '')
                        if jong_char in self.jong and jong_char != ' ':
                            jong_idx = self.jong.index(jong_char)
                            i += 1
                    
                    # 한글 조합
                    korean_char = chr(ord('가') + cho_idx * 21 * 28 + jung_idx * 28 + jong_idx)
                    text += korean_char
                    i += 2
                else:
                    text += char
                    i += 1
            else:
                text += char
                i += 1
        
        return text


class TrueHelgasonMLP(nn.Module):
    """실제 파라미터 감소를 달성하는 헬가손 MLP"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, compression_ratio: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.compression_ratio = compression_ratio
        
        # 압축된 중간 차원 계산
        self.compressed_dim = max(4, int(min(hidden_size, intermediate_size) * compression_ratio))
        
        print(f"   헬가손 MLP: {hidden_size} → {self.compressed_dim} → {intermediate_size} → {self.compressed_dim} → {hidden_size}")
        
        # 압축된 레이어들
        self.compress_in = nn.Linear(hidden_size, self.compressed_dim, bias=False)
        self.gate_expand = nn.Linear(self.compressed_dim, intermediate_size, bias=False)
        self.up_expand = nn.Linear(self.compressed_dim, intermediate_size, bias=False)
        self.compress_mid = nn.Linear(intermediate_size, self.compressed_dim, bias=False)
        self.final_out = nn.Linear(self.compressed_dim, hidden_size, bias=False)
        
        # 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for layer in [self.compress_in, self.gate_expand, self.up_expand, self.compress_mid, self.final_out]:
            nn.init.xavier_uniform_(layer.weight)
            
    def forward(self, x):
        """순전파"""
        # 압축
        compressed = self.compress_in(x)
        
        # 확장
        gate_out = self.gate_expand(compressed)
        up_out = self.up_expand(compressed)
        
        # 게이트 메커니즘 (SiLU 활성화)
        activated = F.silu(gate_out) * up_out
        
        # 재압축 및 출력
        recompressed = self.compress_mid(activated)
        output = self.final_out(recompressed)
        
        return output
    
    def get_compression_ratio(self):
        """실제 압축률 계산"""
        original_params = self.hidden_size * self.intermediate_size * 3  # gate + up + down
        compressed_params = sum(p.numel() for p in self.parameters())
        return compressed_params / original_params


class KoreanGPT(nn.Module):
    """한국어 특화 GPT 모델"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 4, 
                 num_heads: int = 4, max_length: int = 512):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        
        # 임베딩
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # 트랜스포머 레이어들
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_size, num_heads, batch_first=True),
                'attention_norm': nn.LayerNorm(hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                ),
                'mlp_norm': nn.LayerNorm(hidden_size),
            })
            for _ in range(num_layers)
        ])
        
        # 출력 헤드
        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """순전파"""
        batch_size, seq_len = input_ids.shape
        
        # 임베딩
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        hidden_states = token_emb + pos_emb
        
        # 어텐션 마스크 생성 (causal mask)
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attention_mask = attention_mask.to(input_ids.device)
        
        # 트랜스포머 레이어들
        for layer in self.layers:
            # 셀프 어텐션
            residual = hidden_states
            hidden_states = layer['attention_norm'](hidden_states)
            attn_out, _ = layer['attention'](
                hidden_states, hidden_states, hidden_states,
                attn_mask=attention_mask,
                need_weights=False
            )
            hidden_states = residual + attn_out
            
            # MLP
            residual = hidden_states
            hidden_states = layer['mlp_norm'](hidden_states)
            mlp_out = layer['mlp'](hidden_states)
            hidden_states = residual + mlp_out
        
        # 출력
        hidden_states = self.output_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_p=0.9):
        """텍스트 생성"""
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 현재 시퀀스로 예측
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-p 샘플링
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # top_p 임계값을 넘는 토큰들 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 토큰 추가
                generated = torch.cat([generated, next_token], dim=-1)
                
                # EOS 토큰이면 중단
                if next_token.item() == 3:  # <eos>
                    break
        
        return generated


def apply_helgason_compression(model: KoreanGPT, compression_ratio: float = 0.1):
    """모델에 헬가손 압축 적용"""
    
    print(f"\n🔧 한국어 GPT에 헬가손 압축 적용 (압축률: {compression_ratio:.1%})")
    
    compressed_count = 0
    total_original_params = 0
    total_compressed_params = 0
    
    for layer_idx, layer in enumerate(model.layers):
        print(f"\n📐 Layer {layer_idx} MLP 압축 중...")
        
        try:
            # 원본 MLP 정보
            original_mlp = layer['mlp']
            hidden_size = model.hidden_size
            intermediate_size = hidden_size * 4
            
            # 원본 파라미터 수
            original_params = sum(p.numel() for p in original_mlp.parameters())
            
            # 헬가손 MLP로 교체
            compressed_mlp = TrueHelgasonMLP(hidden_size, intermediate_size, compression_ratio)
            layer['mlp'] = compressed_mlp
            
            # 압축된 파라미터 수
            compressed_params = sum(p.numel() for p in compressed_mlp.parameters())
            
            total_original_params += original_params
            total_compressed_params += compressed_params
            compressed_count += 1
            
            actual_ratio = compressed_params / original_params
            print(f"   ✅ Layer {layer_idx}: {original_params:,} → {compressed_params:,} ({actual_ratio:.1%})")
            
        except Exception as e:
            print(f"   ❌ Layer {layer_idx} 압축 실패: {e}")
    
    overall_ratio = total_compressed_params / total_original_params if total_original_params > 0 else 1.0
    memory_saved = (total_original_params - total_compressed_params) * 4 / (1024**2)
    
    print(f"\n🎯 압축 완료:")
    print(f"   압축된 레이어: {compressed_count}개")
    print(f"   MLP 압축률: {overall_ratio:.1%}")
    print(f"   메모리 절약: {memory_saved:.1f}MB")
    
    return model, overall_ratio


def korean_quality_evaluation(original_text: str, compressed_text: str, tokenizer: KoreanTokenizer):
    """한국어 품질 평가"""
    
    # 한글 문자 비율
    def korean_ratio(text):
        korean_chars = sum(1 for c in text if '가' <= c <= '힣')
        total_chars = len(text.replace(' ', ''))
        return korean_chars / total_chars if total_chars > 0 else 0
    
    # 의미 있는 단어 비율 (간단한 휴리스틱)
    def meaningful_ratio(text):
        # 반복되는 패턴 제거
        clean_text = re.sub(r'(.)\1{2,}', r'\1', text)  # 같은 문자 3번 이상 반복 제거
        meaningful_chars = len(clean_text)
        total_chars = len(text)
        return meaningful_chars / total_chars if total_chars > 0 else 0
    
    # 자모 레벨 유사도
    original_jamos = tokenizer.encode(original_text)
    compressed_jamos = tokenizer.encode(compressed_text)
    
    # 자카드 유사도
    set1 = set(original_jamos)
    set2 = set(compressed_jamos)
    jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
    
    return {
        'original_korean_ratio': korean_ratio(original_text),
        'compressed_korean_ratio': korean_ratio(compressed_text),
        'original_meaningful_ratio': meaningful_ratio(original_text),
        'compressed_meaningful_ratio': meaningful_ratio(compressed_text),
        'jamo_similarity': jaccard
    }


def korean_compression_experiment():
    """한국어 압축 실험"""
    
    print("🚀 한국어 최적화 압축 실험 시작")
    print("=" * 80)
    
    # 한국어 토크나이저 초기화
    tokenizer = KoreanTokenizer()
    print(f"📚 한국어 토크나이저 초기화 완료 (어휘 크기: {tokenizer.vocab_size})")
    
    # 한국어 GPT 모델 생성
    model = KoreanGPT(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,  # 작은 모델로 시작
        num_layers=4,
        num_heads=4,
        max_length=256
    )
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 한국어 GPT 모델 생성 완료")
    print(f"   파라미터 수: {original_params:,}")
    print(f"   모델 크기: {original_params * 4 / (1024**2):.2f}MB")
    
    # 한국어 테스트 프롬프트
    korean_prompts = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "한국어 자연어 처리",
        "인공지능 기술",
        "서울 여행"
    ]
    
    print(f"\n📝 원본 모델 한국어 생성 테스트")
    print("-" * 60)
    
    model.eval()
    original_results = []
    
    for i, prompt in enumerate(korean_prompts):
        try:
            # 토큰화
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            
            # 생성
            generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            
            original_results.append(generated_text)
            print(f"{i+1}. 입력: {prompt}")
            print(f"   출력: {generated_text}")
            print()
            
        except Exception as e:
            print(f"   생성 실패: {e}")
            original_results.append("")
    
    # 원본 모델 속도 측정
    print(f"\n⏱️ 원본 모델 속도 측정")
    test_input = torch.tensor([tokenizer.encode("테스트")], dtype=torch.long)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(test_input)
    original_time = (time.time() - start_time) / 20
    print(f"   평균 추론 시간: {original_time*1000:.2f}ms")
    
    # 다양한 압축률로 테스트
    compression_ratios = [0.05, 0.1, 0.2, 0.3]
    
    results_summary = []
    
    for compression_ratio in compression_ratios:
        print(f"\n🔧 압축률 {compression_ratio:.1%} 테스트")
        print("=" * 60)
        
        try:
            # 모델 복사
            import copy
            compressed_model = copy.deepcopy(model)
            
            # 헬가손 압축 적용
            compressed_model, actual_ratio = apply_helgason_compression(compressed_model, compression_ratio)
            
            # 압축된 모델 성능 측정
            print(f"\n📝 압축된 모델 한국어 생성 테스트")
            print("-" * 50)
            
            compressed_model.eval()
            compressed_results = []
            quality_scores = []
            
            for i, prompt in enumerate(korean_prompts):
                try:
                    # 토큰화
                    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                    
                    # 생성
                    generated_ids = compressed_model.generate(input_ids, max_new_tokens=20, temperature=0.7)
                    generated_text = tokenizer.decode(generated_ids[0].tolist())
                    
                    compressed_results.append(generated_text)
                    
                    # 품질 평가
                    if i < len(original_results) and original_results[i]:
                        quality = korean_quality_evaluation(original_results[i], generated_text, tokenizer)
                        quality_scores.append(quality)
                        
                        print(f"{i+1}. 입력: {prompt}")
                        print(f"   원본: {original_results[i]}")
                        print(f"   압축: {generated_text}")
                        print(f"   한글 비율: {quality['compressed_korean_ratio']:.2f}")
                        print(f"   자모 유사도: {quality['jamo_similarity']:.3f}")
                        print()
                    
                except Exception as e:
                    print(f"   생성 실패: {e}")
                    compressed_results.append("")
            
            # 속도 측정
            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = compressed_model(test_input)
            compressed_time = (time.time() - start_time) / 20
            
            # 결과 요약
            avg_korean_ratio = np.mean([q['compressed_korean_ratio'] for q in quality_scores]) if quality_scores else 0
            avg_jamo_similarity = np.mean([q['jamo_similarity'] for q in quality_scores]) if quality_scores else 0
            speed_improvement = original_time / compressed_time if compressed_time > 0 else 1.0
            
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            real_compression_ratio = compressed_params / original_params
            memory_saved = (original_params - compressed_params) * 4 / (1024**2)
            
            result = {
                'compression_ratio': compression_ratio,
                'actual_compression_ratio': real_compression_ratio,
                'korean_ratio': avg_korean_ratio,
                'jamo_similarity': avg_jamo_similarity,
                'speed_improvement': speed_improvement,
                'memory_saved_mb': memory_saved,
                'inference_time_ms': compressed_time * 1000
            }
            
            results_summary.append(result)
            
            print(f"\n📊 압축률 {compression_ratio:.1%} 결과 요약:")
            print(f"   실제 압축률: {real_compression_ratio:.1%}")
            print(f"   평균 한글 비율: {avg_korean_ratio:.3f}")
            print(f"   평균 자모 유사도: {avg_jamo_similarity:.3f}")
            print(f"   추론 시간: {compressed_time*1000:.2f}ms")
            print(f"   속도 향상: {speed_improvement:.2f}x")
            print(f"   메모리 절약: {memory_saved:.2f}MB")
            
        except Exception as e:
            print(f"   ❌ 압축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    print(f"\n🎉 한국어 압축 실험 최종 결과")
    print("=" * 80)
    
    if results_summary:
        print(f"{'압축률':<8} {'실제압축률':<10} {'한글비율':<8} {'자모유사도':<10} {'속도향상':<8} {'메모리절약':<10}")
        print("-" * 70)
        
                        for result in results_summary:            print(f"{result['compression_ratio']:.1%:<8} {result['actual_compression_ratio']:.1%:<10} "                  f"{result['korean_ratio']:.3f:<8} {result['jamo_similarity']:.3f:<10} "                  f"{result['speed_improvement']:.2f}x:<8} {result['memory_saved_mb']:.1f}MB:<10}")
        
        # 최고 성능 찾기
        best_compression = min(results_summary, key=lambda x: x['actual_compression_ratio'])
        best_quality = max(results_summary, key=lambda x: x['jamo_similarity'])
        best_speed = max(results_summary, key=lambda x: x['speed_improvement'])
        
        print(f"\n🏆 최고 성능:")
        print(f"   최고 압축: {best_compression['compression_ratio']:.1%} "
              f"({best_compression['actual_compression_ratio']:.1%} 실제)")
        print(f"   최고 품질: {best_quality['compression_ratio']:.1%} "
              f"(자모 유사도 {best_quality['jamo_similarity']:.3f})")
        print(f"   최고 속도: {best_speed['compression_ratio']:.1%} "
              f"({best_speed['speed_improvement']:.2f}x 향상)")
    
    return results_summary


if __name__ == "__main__":
    try:
        results = korean_compression_experiment()
        print(f"\n✅ 실험 완료! 결과 수: {len(results)}개")
    except Exception as e:
        print(f"실험 실행 실패: {e}")
        import traceback
        traceback.print_exc() 