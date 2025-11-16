import torch
from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalLLMConfig,
    train_hierarchical_llm_from_text,
)


def main():
    print("=" * 70)
    print("Reality Stone - 실제 데이터 학습 및 테스트")
    print("=" * 70)
    
    data_path = "data/text.txt"
    
    tokenizer_name = "klue/bert-base"
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = len(tokenizer)
        print(f"토크나이저 로드: {tokenizer_name}, vocab_size={vocab_size}")
    except Exception as e:
        print(f"토크나이저 로드 실패: {e}")
        vocab_size = 32000
    
    config = HierarchicalLLMConfig(
        vocab_size=vocab_size,
        d_model=128,
        d_head=32,
        num_topics=8,
        num_heads_topic=2,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        pretrained_tokenizer=tokenizer_name,
        lr_backbone=3e-4,
        lr_metric=1e-4,
        lambda_consistency=0.02,
        lambda_diversity=0.01,
        lambda_consistency_schedule="warmup",
        lambda_diversity_schedule="warmup",
        lambda_semantic=0.0,
        lambda_metric=0.0,
        lambda_length=0.0,
        grad_clip_norm=1.0,
        logit_clip_value=20.0,
        loss_clip_max=100.0,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n사용 디바이스: {device}")
    print(f"데이터 경로: {data_path}")
    print(f"\n모델 설정 (메모리 최적화):")
    print(f"  - Vocab Size: {config.vocab_size}")
    print(f"  - Model Dim: {config.d_model}")
    print(f"  - Num Topics: {config.num_topics}")
    print(f"  - Decoder Layers: {config.n_layer_decoder}")
    
    print("\n" + "=" * 70)
    print("1. 모델 학습")
    print("=" * 70)
    
    model, info = train_hierarchical_llm_from_text(
        data_path=data_path,
        config=config,
        epochs=5,
        batch_size=2,
        max_paragraphs=200,
        device=device,
    )
    
    print(f"\n학습 완료!")
    print(f"  - 최종 Loss: {info['final_loss']:.4f}")
    print(f"  - 샘플 수: {info['num_samples']}")
    print(f"  - 총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    llm = HierarchicalLLM(model, config, torch.device(device))
    
    print("\n" + "=" * 70)
    print("2. 텍스트 생성 테스트")
    print("=" * 70)
    
    test_prompts = [
        "인공지능은",
        "딥러닝 모델의 구조는",
        "자연어 처리에서 중요한 것은",
    ]
    
    generator = pipeline("text-generation", model=llm, max_length=48, k_neighbors=3)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[테스트 {i}]")
        print(f"입력: {prompt}")
        
        result = generator(prompt, max_new_tokens=20, return_dict=True)
        
        print(f"생성: {result['generated_text']}")
        
        if result['topics']:
            top_topics = result['topics'][:2]
            print(f"주제:")
            for topic in top_topics:
                print(f"  - {topic['topic']} (신뢰도: {topic['confidence']:.3f})")
    
    print("\n" + "=" * 70)
    print("3. 텍스트 편집 테스트")
    print("=" * 70)
    
    editor = pipeline("text-editing", model=llm, max_length=48, enable_structural_edit=False)
    
    edit_texts = [
        "이것은 테스트 문장입니다.",
        "기계학습은 데이터로부터 패턴을 학습합니다.",
    ]
    
    for i, text in enumerate(edit_texts, 1):
        print(f"\n[편집 {i}]")
        print(f"원본: {text}")
        
        result = editor(text, return_topics=True)
        
        print(f"편집: {result['edited']}")
        
        if result['topics']:
            print(f"주제: {result['topics'][0]['topic']}")
    
    print("\n" + "=" * 70)
    print("4. 질의응답 테스트")
    print("=" * 70)
    
    qa = pipeline(
        "question-answering",
        model=llm,
        corpus=data_path,
        top_k=5,
        use_llm=False,
    )
    
    questions = [
        "인공지능이란 무엇인가?",
        "딥러닝의 특징은?",
        "자연어 처리는 어떻게 작동하는가?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[질문 {i}] {question}")
        
        answer = qa(question)
        
        if 'answers' in answer and answer['answers']:
            print(f"답변:")
            for rank, ans in enumerate(answer['answers'][:3], 1):
                sentence = ans['sentence'][:80] + "..." if len(ans['sentence']) > 80 else ans['sentence']
                print(f"  {rank}. {sentence}")
                if not torch.isnan(torch.tensor(ans['distance'])):
                    print(f"     (거리: {ans['distance']:.4f})")
    
    print("\n" + "=" * 70)
    print("5. 문서 검색 테스트")
    print("=" * 70)
    
    indexer = pipeline("document-indexing", model=llm, max_paragraphs=200)
    
    print("\n인덱스 구축 중...")
    index = indexer.build_index(data_path)
    print(f"인덱싱 완료: {len(index)}개 문장")
    
    search_queries = [
        "머신러닝 알고리즘",
        "신경망 학습",
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n[검색 {i}] {query}")
        
        results = indexer.search(query, top_k=3)
        
        print(f"검색 결과:")
        for r in results:
            sentence = r['sentence'][:80] + "..." if len(r['sentence']) > 80 else r['sentence']
            print(f"  [{r['rank']}] {sentence}")
            if not torch.isnan(torch.tensor(r['distance'])):
                print(f"       (거리: {r['distance']:.4f})")
    
    print("\n" + "=" * 70)
    print("전체 테스트 완료!")
    print("=" * 70)
    
    model_save_path = "checkpoints/reality_stone_model.pt"
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'info': info,
    }, model_save_path)
    print(f"\n모델 저장됨: {model_save_path}")


if __name__ == "__main__":
    main()

