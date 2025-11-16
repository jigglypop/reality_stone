import tempfile
from pathlib import Path
import torch

from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalLLMConfig,
    train_hierarchical_llm_from_text,
)


def create_test_corpus():
    corpus_content = """환자는 65세 남성으로 흉통을 주소로 내원하였습니다.

과거력상 고혈압과 당뇨병이 있습니다. 현재 약물 복용 중입니다.

신체검사에서 혈압 140/90, 맥박 82회/분으로 측정되었습니다.

심전도 검사 결과 정상 소견을 보였습니다.

흉부 X-ray에서 특이 소견은 관찰되지 않았습니다.

최종 진단은 비특이적 흉통으로, 경과 관찰하기로 하였습니다.

외래 추적 관찰 예정입니다.

환자는 처방받은 약물을 규칙적으로 복용하고 있습니다.

혈압 조절이 잘 되고 있으며 증상 호전되었습니다.
"""
    
    tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
    tmpfile.write(corpus_content)
    tmpfile.close()
    return tmpfile.name


def test_training():
    print("=" * 60)
    print("1. 모델 학습 테스트")
    print("=" * 60)
    
    corpus_path = create_test_corpus()
    
    config = HierarchicalLLMConfig(
        vocab_size=500,
        d_model=64,
        d_head=16,
        num_topics=4,
        num_heads_topic=2,
        n_layer_decoder=1,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    
    print("학습 시작...")
    model, info = train_hierarchical_llm_from_text(
        data_path=corpus_path,
        config=config,
        epochs=2,
        batch_size=2,
        max_paragraphs=10,
        device='cpu',
    )
    
    print(f"학습 완료!")
    print(f"- 최종 Loss: {info['final_loss']:.4f}")
    print(f"- 샘플 수: {info['num_samples']}")
    
    return model, corpus_path


def test_inference(model):
    print("\n" + "=" * 60)
    print("2. 텍스트 생성 테스트")
    print("=" * 60)
    
    llm = HierarchicalLLM(model, model.config, torch.device('cpu'))
    
    test_text = "환자는 고혈압 진단을 받았습니다."
    print(f"\n입력: {test_text}")
    
    result = llm.generate(test_text, max_length=32, return_dict=True)
    
    print(f"\n생성 결과:")
    print(f"- 원본: {result['original_text']}")
    print(f"- 생성: {result['generated_text']}")
    print(f"- 문장 수: {len(result['sentences'])}")
    print(f"- 주제 수: {len(result['topics'])}")
    
    if result['topics']:
        print(f"\n주제 정보:")
        for i, topic in enumerate(result['topics'][:3]):
            print(f"  [{i+1}] {topic['sentence']}")
            print(f"      주제: {topic['topic']}, 신뢰도: {topic['confidence']:.3f}")
    
    return llm


def test_pipeline_generation(llm):
    print("\n" + "=" * 60)
    print("3. Pipeline API 테스트 (텍스트 생성)")
    print("=" * 60)
    
    generator = pipeline("text-generation", model=llm, max_length=32)
    
    test_texts = [
        "진단 결과 정상입니다.",
        "혈압이 높아서 약을 처방받았습니다.",
    ]
    
    print("\n배치 생성:")
    outputs = generator(test_texts)
    
    for i, (inp, out) in enumerate(zip(test_texts, outputs)):
        print(f"\n[{i+1}] 입력: {inp}")
        print(f"    출력: {out}")


def test_text_editing(llm):
    print("\n" + "=" * 60)
    print("4. 텍스트 편집 테스트")
    print("=" * 60)
    
    editor = pipeline("text-editing", model=llm, max_length=32, enable_structural_edit=False)
    
    test_text = "환자의 상태가 호전되었습니다."
    print(f"\n입력: {test_text}")
    
    result = editor(test_text, return_topics=True)
    
    print(f"\n편집 결과:")
    print(f"- 원본: {result['original']}")
    print(f"- 편집: {result['edited']}")
    print(f"- 주제 개수: {len(result['topics'])}")


def test_qa(llm, corpus_path):
    print("\n" + "=" * 60)
    print("5. 질의응답 테스트")
    print("=" * 60)
    
    qa = pipeline(
        "question-answering",
        model=llm,
        corpus=corpus_path,
        top_k=3,
        use_llm=False,
    )
    
    questions = [
        "환자의 혈압은 얼마인가요?",
        "진단 결과는 무엇인가요?",
    ]
    
    print("\n질의응답 결과:")
    for i, question in enumerate(questions):
        print(f"\n[{i+1}] 질문: {question}")
        
        answer = qa(question)
        
        if 'answers' in answer and answer['answers']:
            print(f"     답변 문장:")
            for rank, ans in enumerate(answer['answers'][:2], 1):
                print(f"       {rank}. {ans['sentence']}")
                print(f"          (거리: {ans['distance']:.4f})")


def test_indexing(llm, corpus_path):
    print("\n" + "=" * 60)
    print("6. 문서 인덱싱 및 검색 테스트")
    print("=" * 60)
    
    indexer = pipeline("document-indexing", model=llm, max_paragraphs=10)
    
    print("\n인덱스 구축 중...")
    index = indexer.build_index(corpus_path)
    print(f"인덱싱 완료: {len(index)}개 문장")
    
    query = "혈압 측정 결과"
    print(f"\n검색 쿼리: {query}")
    
    results = indexer.search(query, top_k=3)
    
    print(f"\n검색 결과:")
    for r in results:
        print(f"  [{r['rank']}] {r['sentence']}")
        print(f"       거리: {r['distance']:.4f}")


def main():
    print("\n" + "=" * 60)
    print("Reality Stone API 전체 파이프라인 테스트")
    print("=" * 60)
    
    model, corpus_path = test_training()
    
    llm = test_inference(model)
    
    test_pipeline_generation(llm)
    
    test_text_editing(llm)
    
    test_qa(llm, corpus_path)
    
    test_indexing(llm, corpus_path)
    
    print("\n" + "=" * 60)
    print("전체 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

