#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
추론 스크립트 - 학습된 모델로 실제 응답 생성
"""

import sys
import io
from pathlib import Path

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / "python"))

import torch
import json
from datetime import datetime

from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.models.sentence_topic_head import SentenceTopicHead
from reality_stone.models.metric_router import MetricContextRouter
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder


def load_model(checkpoint_path='checkpoint_best.pt'):
    """체크포인트에서 모델 로드"""
    print(f"📦 체크포인트 로딩: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 모델 초기화
    topic_head = SentenceTopicHead(
        d_model=config['d_model'],
        d_head=config['d_head'],
        num_topics=config['num_topics'],
        num_heads=config['num_heads']
    )
    topic_head.load_state_dict(checkpoint['topic_head'])
    topic_head.eval()
    
    router = MetricContextRouter(d_head=config['d_head'])
    
    decoder = RCELexicalDecoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        n_head=config['n_head']
    )
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    
    print(f"✓ 모델 로드 완료 (epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")
    
    return topic_head, router, decoder, config


def inference(text, topic_head, router, decoder, config, segmenter):
    """추론 실행"""
    device = next(topic_head.parameters()).device
    
    # L0: Pre-Segmenter
    seg_output = segmenter(text)
    
    if len(seg_output['sentences']) == 0:
        return {
            'error': '문장을 분해할 수 없습니다.',
            'original': text
        }
    
    print(f"\n📝 입력 분석:")
    print(f"  - 원문: {text[:100]}...")
    print(f"  - 문장 수: {len(seg_output['sentences'])}")
    for i, sent in enumerate(seg_output['sentences'], 1):
        print(f"    {i}. {sent}")
    
    # 배치 차원 추가
    tokens = seg_output['tokens'].unsqueeze(0).to(device)  # [1, T, seq_len]
    topo_idx = seg_output['topo_idx'].unsqueeze(0).to(device)
    mask = seg_output['replacement_mask'].unsqueeze(0).to(device)
    
    T, seq_len = seg_output['tokens'].shape
    
    # 첫 번째 토큰만 사용
    tokens_input = tokens[:, :, 0]  # [1, T]
    tokens_input = torch.clamp(tokens_input, 0, decoder.vocab_size - 1)
    mask_input = mask[:, :, 0]  # [1, T]
    
    # 문장 임베딩
    sentence_emb = tokens.float().mean(dim=2)  # [1, T]
    sentence_emb = sentence_emb.unsqueeze(-1).expand(-1, -1, 768)  # [1, T, 768]
    
    with torch.no_grad():
        # L1: SentenceTopicHead (SPD Metric Learning!)
        P_topic, scores, metric_keys = topic_head(sentence_emb, topo_idx)
        
        print(f"\n🎯 주제 분류 (SPD Metric Learning):")
        topic_names = topic_head.topic_names
        for i, (prob, key) in enumerate(zip(P_topic[0], metric_keys), 1):
            top_topic_idx = prob.argmax().item()
            top_prob = prob[top_topic_idx].item()
            print(f"  문장 {i}: {topic_names[top_topic_idx]} ({top_prob:.2%})")
            print(f"           Metric Key: {key}")
        
        # L2: MetricContextRouter
        L_i = router(metric_keys, scores)
        print(f"\n🔧 SPD 메트릭 생성: {L_i.shape}")
        
        # L3: RCE-LexicalDecoder
        candidates = {
            int(tid): [int(tid), min(int(tid)+1, decoder.vocab_size-1), min(int(tid)+2, decoder.vocab_size-1)]
            for tid in torch.unique(tokens_input).tolist() if tid > 0
        }
        
        output_ids, logits = decoder(tokens_input, L_i, mask_input, topo_idx, candidates)
        
        # 변경 통계
        original_tokens = seg_output['tokens'][:, 0]
        changed = (output_ids[0].cpu() != original_tokens).sum().item()
        total = len(original_tokens)
        
        print(f"\n📊 재작성 결과:")
        print(f"  - 변경된 토큰: {changed}/{total} ({changed/total*100:.1f}%)")
        
        # 결과 구성
        result = {
            'original_text': text,
            'sentences': seg_output['sentences'],
            'num_sentences': len(seg_output['sentences']),
            'topics': [
                {
                    'sentence': sent,
                    'topic': topic_names[P_topic[0][i].argmax().item()],
                    'confidence': P_topic[0][i].max().item(),
                    'metric_key': metric_keys[i],
                    'all_probs': {
                        topic_names[j]: P_topic[0][i][j].item()
                        for j in range(len(topic_names))
                    }
                }
                for i, sent in enumerate(seg_output['sentences'])
            ],
            'tokens_changed': changed,
            'tokens_total': total,
            'change_ratio': changed / total,
            'output_token_ids': output_ids[0].cpu().tolist()
        }
        
        return result


def interactive_mode(topic_head, router, decoder, config):
    """대화형 모드"""
    segmenter = PreSegmenter(max_length=config['max_length'])
    
    print("\n" + "=" * 70)
    print("🤖 대화형 추론 모드")
    print("=" * 70)
    print("\n입력할 텍스트를 입력하세요 (종료: 'quit' 또는 Ctrl+C)")
    print()
    
    while True:
        try:
            text = input("📝 입력> ").strip()
            
            if not text:
                continue
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 종료합니다.")
                break
            
            # 추론 실행
            result = inference(text, topic_head, router, decoder, config, segmenter)
            
            if 'error' in result:
                print(f"\n❌ {result['error']}")
                continue
            
            # 결과 출력
            print(f"\n✅ 분석 완료!")
            print(f"\n📋 상세 결과:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("\n" + "-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()


def batch_test():
    """배치 테스트"""
    test_cases = [
        "양자역학은 현대 물리학의 핵심이다. 미시 세계를 설명하는 이론이다. 많은 실험으로 검증되었다.",
        "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다. 정기적인 검진이 필요하다.",
        "인공지능 기술이 발전하고 있다. 의료 분야에 적용되고 있다. 진단 정확도가 향상되었다.",
        "한국의 역사는 매우 오래되었다. 삼국시대부터 현대까지 이어진다. 문화유산이 풍부하다.",
        "기후 변화가 심각하다. 온실가스 배출을 줄여야 한다. 국제적 협력이 필요하다."
    ]
    
    print("\n" + "=" * 70)
    print("🧪 배치 테스트")
    print("=" * 70)
    
    # 모델 로드
    topic_head, router, decoder, config = load_model()
    segmenter = PreSegmenter(max_length=config['max_length'])
    
    results = []
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {i}/{len(test_cases)}")
        print(f"{'='*70}")
        
        result = inference(text, topic_head, router, decoder, config, segmenter)
        results.append(result)
    
    # 결과 저장
    output_file = f'inference_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    # 통계
    print(f"\n📊 전체 통계:")
    print(f"  - 총 테스트: {len(results)}개")
    print(f"  - 평균 문장 수: {sum(r['num_sentences'] for r in results) / len(results):.1f}")
    print(f"  - 평균 변경률: {sum(r['change_ratio'] for r in results) / len(results) * 100:.1f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentence-Topic LLM 추론')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='batch',
                       help='실행 모드 (interactive: 대화형, batch: 배치 테스트)')
    parser.add_argument('--checkpoint', default='checkpoint_best.pt',
                       help='체크포인트 경로')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        topic_head, router, decoder, config = load_model(args.checkpoint)
        interactive_mode(topic_head, router, decoder, config)
    else:
        batch_test()


if __name__ == "__main__":
    main()

