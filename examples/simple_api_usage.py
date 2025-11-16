from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import HierarchicalLLMConfig


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
print(result["generated_text"])

result_dict = model.generate(text, max_length=64, return_dict=True)
print(result_dict)

generator = pipeline("text-generation", config=config, max_length=64, k_neighbors=2)
output = generator(text)
print(output)

output_dict = generator(text, return_dict=True, max_new_tokens=15)
print(output_dict)

batch_texts = ["첫 번째 문장", "두 번째 문장"]
batch_outputs = generator(batch_texts)
print(batch_outputs)

editor = pipeline("text-editing", config=config, enable_structural_edit=True)
edited = editor(text)
print(edited)

edited_simple = editor(text, return_topics=False, max_length=32)
print(edited_simple)

