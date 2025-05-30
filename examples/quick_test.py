#!/usr/bin/env python3
"""
빠른 극한 압축 모델 테스트
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
from collections import Counter

def generate_with_anti_repetition(model, tokenizer, prompt, max_length=25):
    """극한 반복 방지 생성 (한국어 초특화)"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.6,          # 보수적 온도
            top_p=0.8,               # 제한적 확률 
            top_k=30,                # 제한적 선택
            repetition_penalty=1.8,   # 반복 페널티 극대화
            no_repeat_ngram_size=5,   # n-gram 크기 확대
            pad_token_id=tokenizer.eos_token_id,
            # beam search 관련 설정들 제거 (충돌 해결)
            min_length=len(inputs.input_ids[0]) + 2,  # 최소 길이 보장
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def advanced_quality_evaluation(generated_text, prompt):
    """엄격한 한국어 품질 평가 시스템"""
    
    generated_only = generated_text[len(prompt):].strip()
    if len(generated_only) < 2:
        return 0.0
    
    score = 0.0
    max_score = 7.0
    
    # 1. 반복 패턴 검사 (0-2점)
    repetition_penalty = calculate_repetition_penalty(generated_only)
    repetition_score = max(0, 2.0 - repetition_penalty * 4)
    score += repetition_score
    
    # 2. 한국어 문법 구조 (0-2점)
    grammar_score = evaluate_korean_grammar(generated_only)
    score += grammar_score
    
    # 3. 의미 연관성 (0-1.5점)
    semantic_score = calculate_semantic_relevance(prompt, generated_only)
    score += semantic_score * 1.5
    
    # 4. 텍스트 자연스러움 (0-1점)
    naturalness_score = evaluate_naturalness(generated_only)
    score += naturalness_score
    
    # 5. 특수문자/오류 패턴 페널티 (0-0.5점)
    error_penalty = calculate_error_penalty(generated_only)
    score += max(0, 0.5 - error_penalty)
    
    return min(score / max_score * 3.0, 3.0)

def calculate_repetition_penalty(text):
    """반복 패턴 페널티 계산"""
    char_repeats = len(re.findall(r'(.)\1{2,}', text))
    words = text.split()
    if len(words) > 1:
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 2)
    else:
        repeated_words = 0
    punct_repeats = len(re.findall(r'[.!?]{3,}|[~]{2,}|[/]{2,}', text))
    total_penalty = min(1.0, (char_repeats + repeated_words + punct_repeats * 2) / 10)
    return total_penalty

def evaluate_korean_grammar(text):
    """한국어 문법 구조 평가"""
    score = 0.0
    korean_endings = ['다', '요', '니다', '해요', '어요', '아요', '네요', '죠', '습니다', '겠습니다']
    has_proper_ending = any(text.endswith(ending) for ending in korean_endings)
    if has_proper_ending:
        score += 1.0
    elif any(ending in text for ending in korean_endings):
        score += 0.5
    
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    if sentences:
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 2)
        if complete_sentences > 0:
            score += 0.8
        else:
            score += 0.3
    
    particles = ['이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의']
    has_particles = any(p in text for p in particles)
    if has_particles:
        score += 0.2
    
    return min(score, 2.0)

def evaluate_naturalness(text):
    """텍스트 자연스러움 평가"""
    score = 1.0
    weird_patterns = [
        r'[.]{3,}', r'[!]{2,}', r'[?]{2,}', r'[/]{2,}', 
        r'[~]{3,}', r'[:]{2,}', r'[0-9]{5,}'
    ]
    
    for pattern in weird_patterns:
        if re.search(pattern, text):
            score -= 0.3
    
    words = text.split()
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length > 10:
            score -= 0.3
    
    return max(0, score)

def calculate_error_penalty(text):
    """오류 패턴 페널티 계산"""
    penalty = 0.0
    severe_errors = [
        r'[가-힣]+[/]+[가-힣]+', r'[:-]+[/]+',
        r'[&+-]{2,}', r'[()\[\]]{3,}'
    ]
    
    for pattern in severe_errors:
        matches = len(re.findall(pattern, text))
        penalty += matches * 0.5
    
    return penalty

def calculate_semantic_relevance(prompt, generated):
    """의미적 연관성 계산"""
    keyword_mapping = {
        '안녕': ['안녕', '반갑', '좋', '감사'],
        '날씨': ['날씨', '맑', '흐림', '비', '눈', '따뜻', '춥', '좋'],
        '수도': ['서울', '도시', '한국', '수도'],
        '인공지능': ['AI', '기술', '컴퓨터', '로봇', '지능', '학습'],
        '음식': ['음식', '맛', '먹', '요리', '식사'],
    }
    
    relevance = 0.0
    for key, keywords in keyword_mapping.items():
        if key in prompt:
            matches = sum(1 for kw in keywords if kw in generated)
            relevance = max(relevance, min(1.0, matches / 2))
    
    return relevance

def quick_test():
    """빠른 테스트"""
    
    print("🚀 극한 압축 모델 빠른 생성 테스트")
    print("=" * 60)
    
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_prompts = [
        "안녕하세요",
        "오늘 날씨는", 
        "한국의 수도는",
        "인공지능이란",
        "맛있는 음식은"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/5] '{prompt}'")
        
        try:
            t0 = time.time()
            generated_text = generate_with_anti_repetition(model, tokenizer, prompt, max_length=25)
            elapsed = time.time() - t0
            
            print(f"  생성: {generated_text}")
            print(f"  시간: {elapsed:.3f}초")
            
            quality_score = advanced_quality_evaluation(generated_text, prompt)
            print(f"  품질: {quality_score:.2f}/3.0")
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'time': elapsed,
                'quality': quality_score
            })
            
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            results.append({
                'prompt': prompt,
                'generated': f"ERROR: {e}",
                'time': 0,
                'quality': 0
            })
    
    # 통계
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 빠른 테스트 통계:")
    print(f"  평균 시간: {avg_time:.3f}초")
    print(f"  평균 품질: {avg_quality:.2f}/3.0")
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 모델 정보:")
    print(f"  파라미터 수: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  모델 타입: 원본 모델 (압축 테스트용)")
    
    if avg_quality >= 2.0:
        print("✅ 생성 기능 정상 작동!")
    elif avg_quality >= 1.0:
        print("🔧 생성 기능 부분 작동")
    else:
        print("❌ 생성 기능 문제 있음")

if __name__ == "__main__":
    quick_test()