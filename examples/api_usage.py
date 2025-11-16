import torch
from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import HierarchicalLLMConfig


def example_text_generation():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    model = HierarchicalLLM.from_config(config)
    
    text = "환자는 고혈압 진단을 받았습니다."
    result = model(text)
    
    print(f"Original: {result['original_text']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Topics: {result['topics']}")


def example_pipeline_generation():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    generator = pipeline("text-generation", config=config)
    
    text = "진단 결과 정상입니다."
    output = generator(text)
    
    print(f"Generated: {output}")


def example_text_editing():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    editor = pipeline("text-editing", config=config)
    
    text = "혈압이 높아서 약을 처방받았습니다."
    result = editor(text, enable_structural_edit=True)
    
    print(f"Original: {result['original']}")
    print(f"Edited: {result['edited']}")
    print(f"Topics: {result['topics']}")


def example_question_answering():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    qa = pipeline(
        "question-answering",
        config=config,
        corpus="path/to/corpus.txt",
        use_llm=True
    )
    
    question = "환자의 진단 결과는 무엇인가요?"
    answer = qa(question)
    
    print(f"Question: {answer['question']}")
    print(f"Answer: {answer['answer']}")
    print(f"Support: {answer['support']}")


def example_document_indexing():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    indexer = pipeline("document-indexing", config=config)
    
    index = indexer.build_index("path/to/corpus.txt", max_paragraphs=100)
    
    print(f"Indexed {len(index)} sentences")
    
    results = indexer.search("환자의 혈압", top_k=3)
    
    for r in results:
        print(f"Rank {r['rank']}: {r['sentence']} (distance: {r['distance']:.4f})")


def example_save_load():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        use_pretrained_embeddings=False,
    )
    
    model = HierarchicalLLM.from_config(config)
    
    model.save_pretrained("./saved_model")
    
    loaded_model = HierarchicalLLM.from_pretrained("./saved_model/model.pt")
    
    text = "테스트 문장입니다."
    result = loaded_model(text)
    
    print(f"Generated: {result['generated_text']}")


if __name__ == "__main__":
    print("=== Text Generation ===")
    example_text_generation()
    
    print("\n=== Pipeline Generation ===")
    example_pipeline_generation()
    
    print("\n=== Text Editing ===")
    example_text_editing()

