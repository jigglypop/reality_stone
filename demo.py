#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence-Topic LLM 데모
docs/sentence_topic_architecture.md 명세 구현 검증
"""

import sys
import io

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=" * 70)
    print("Sentence-Topic LLM - 실행 데모")
    print("=" * 70)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "python"))
    
    # Phase 1: Pre-Segmenter
    print("\n[1/5] Pre-Segmenter 로딩...")
    from reality_stone.utils.pre_segmenter import PreSegmenter
    segmenter = PreSegmenter(max_length=128, k_neighbors=3)
    print("✓ Pre-Segmenter 로드 완료")
    
    # Phase 2: SentenceTopicHead
    print("\n[2/5] SentenceTopicHead 로딩...")
    from reality_stone.models.sentence_topic_head import SentenceTopicHead
    topic_head = SentenceTopicHead(d_model=768, d_head=64, num_topics=8, num_heads=4)
    topic_head.eval()
    print("✓ SentenceTopicHead 로드 완료")
    
    # Phase 3: MetricContextRouter
    print("\n[3/5] MetricContextRouter 로딩...")
    from reality_stone.models.metric_router import MetricContextRouter
    router = MetricContextRouter(d_head=64)
    print("✓ MetricContextRouter 로드 완료")
    
    # Phase 4: RCE-LexicalDecoder
    print("\n[4/5] RCE-LexicalDecoder 로딩...")
    from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder
    decoder = RCELexicalDecoder(vocab_size=50000, d_model=768, n_layer=4, n_head=8)
    decoder.eval()
    print("✓ RCE-LexicalDecoder 로드 완료")
    
    # 테스트 실행
    print("\n[5/5] 파이프라인 테스트 실행...")
    print("-" * 70)
    
    test_cases = [
        "양자역학은 모든 역학을 포함한다. 고전역학으로 설명되지 않는 현상을 설명한다.",
        "환자는 고혈압 진단을 받았다. 약물 치료를 시작했다. 경과가 양호하다.",
        "중국의 역사는 매우 길다. 여러 왕조가 흥망을 반복했다."
    ]
    
    import torch
    
    for idx, paragraph in enumerate(test_cases, 1):
        print(f"\n테스트 {idx}:")
        print(f"입력: {paragraph[:50]}...")
        
        try:
            # L0: Pre-Segmenter
            seg_output = segmenter(paragraph)
            num_sentences = len(seg_output['sentences'])
            print(f"  ✓ L0: {num_sentences}개 문장 분해")
            
            if num_sentences == 0:
                print("  (빈 결과 - 스킵)")
                continue
            
            # L1: SentenceTopicHead
            T, seq_len = seg_output['tokens'].shape
            sentence_emb = seg_output['tokens'].float().mean(dim=1)  # [T]
            sentence_emb = sentence_emb.unsqueeze(-1).expand(-1, 768)  # [T, 768]
            sentence_emb = sentence_emb.unsqueeze(0)  # [1, T, 768]
            
            topo_input = seg_output['topo_idx'].unsqueeze(0)  # [1, T, K]
            
            with torch.no_grad():
                P_topic, scores, metric_keys = topic_head(sentence_emb, topo_input)
            
            print(f"  ✓ L1: 주제 분류 완료")
            print(f"       주제: {[f'{p.max():.2f}' for p in P_topic[0]]}")
            print(f"       Keys: {metric_keys[:2]}...")
            
            # L2: MetricContextRouter
            L_i = router(metric_keys, scores)
            print(f"  ✓ L2: SPD 메트릭 생성 ({L_i.shape})")
            
            # L3: RCE-LexicalDecoder (간소화)
            # tokens는 [T, seq_len]이므로 첫 번째 토큰만 사용
            tokens_input = seg_output['tokens'][:, 0].unsqueeze(0)  # [1, T]
            # vocab_size 범위 내로 클램핑
            tokens_input = torch.clamp(tokens_input, 0, decoder.vocab_size - 1)
            # replacement_mask도 첫 번째 토큰만
            mask_input = seg_output['replacement_mask'][:, 0].unsqueeze(0)  # [1, T]
            
            # 간단한 후보 생성 (vocab_size 범위 내)
            unique_tokens = torch.unique(tokens_input)
            candidates = {
                int(tid): [int(tid), min(int(tid)+1, decoder.vocab_size-1), min(int(tid)+2, decoder.vocab_size-1)] 
                for tid in unique_tokens.tolist() if tid > 0
            }
            
            with torch.no_grad():
                output_ids, _ = decoder(
                    tokens_input, L_i, mask_input, topo_input, candidates
                )
            
            # 변경 통계 (첫 번째 토큰만 비교)
            original_tokens = seg_output['tokens'][:, 0]
            changed = (output_ids[0] != original_tokens).sum().item()
            total = len(original_tokens)
            
            print(f"  ✓ L3: 디코더 실행 완료")
            print(f"       변경: {changed}/{total} 토큰 ({changed/total*100:.1f}%)")
            
            print(f"  ✓✓ 테스트 {idx} 성공!")
            
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("데모 완료!")
    print("\nAPI 서버 실행:")
    print("  python api/server.py")
    print("\nAPI 테스트:")
    print("  python tests/test_sentence_topic_api.py")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n중단됨")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

