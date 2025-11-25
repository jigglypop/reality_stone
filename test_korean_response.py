import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python"))

import torch
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    infer_hierarchical_llm_on_text,
)

def test_korean_response():
    print("=" * 80)
    print("한글 응답 테스트 시작")
    print("=" * 80)
    
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
        max_answer_sentences=20,
        lambda_consistency=0.5,
        lambda_diversity=0.1,
        lambda_topic_supervision=0.5,
        lambda_semantic=0.3,
        max_lm_seq_len=1024,
    )
    
    print(f"\n[Config]")
    print(f"  max_answer_sentences: {config.max_answer_sentences}")
    print(f"  max_lm_seq_len: {config.max_lm_seq_len}")
    print(f"  lambda_consistency: {config.lambda_consistency}")
    print(f"  lambda_topic_supervision: {config.lambda_topic_supervision}")
    print(f"  lambda_semantic: {config.lambda_semantic}")
    
    model = HierarchicalSentenceTopicLLM(config)
    model.eval()
    
    test_texts = [
        "안녕하세요. 오늘 날씨가 좋네요. 산책하기 좋은 날입니다.",
        "인공지능 기술이 발전하고 있습니다. 특히 자연어 처리 분야가 주목받고 있습니다.",
        "서울은 대한민국의 수도입니다. 인구가 천만 명이 넘는 대도시입니다.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'=' * 80}")
        print(f"테스트 {i}/{len(test_texts)}")
        print(f"{'=' * 80}")
        print(f"[입력 텍스트]")
        print(f"  {text}")
        
        try:
            with torch.no_grad():
                result = infer_hierarchical_llm_on_text(
                    model=model,
                    text=text,
                    max_length=128,
                    k_neighbors=3,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.9,
                    use_sampling=True,
                    use_top_down=False,
                )
            
            print(f"\n[결과]")
            print(f"  원본 문장 수: {len(result.get('sentences', []))}")
            print(f"  생성된 텍스트 길이: {len(result.get('generated_text', ''))}")
            print(f"\n[생성된 텍스트]")
            print(f"  {result.get('generated_text', '(없음)')}")
            
            if result.get('topics'):
                print(f"\n[주제 정보]")
                for j, topic_info in enumerate(result['topics'][:3], 1):
                    print(f"  문장 {j}: {topic_info.get('topic', 'N/A')} "
                          f"(신뢰도: {topic_info.get('confidence', 0):.3f})")
            
            print(f"\n[성공] 응답 생성 완료")
            
        except Exception as e:
            import traceback
            print(f"\n[ERROR] 응답 생성 실패")
            print(f"  오류: {e}")
            print(f"\n[상세 오류 정보]")
            print(traceback.format_exc())
    
    print(f"\n{'=' * 80}")
    print("테스트 완료")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_korean_response()


