import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from reality_stone.metric_extraction import extract_riemannian_metric
import time

def train_extracted_metric():
    print("=== Reality Stone: Metric Fine-Tuning (Resurrecting Intelligence) ===")
    
    # 1. 모델 준비
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. 메트릭 추출 (94% 압축 상태로 시작)
    # 초기에는 뇌가 텅 빈 상태(붕괴된 언어)입니다.
    model = extract_riemannian_metric(model, target_dim=64) # 32는 너무 작으니 64로 시도
    
    # 3. 학습 가능한 파라미터 확인
    # 대부분의 파라미터(기저 벡터)는 고정하고, 오직 '메트릭 텐서(G)'만 학습할 수도 있지만,
    # 여기서는 빠른 복구를 위해 기저 벡터(U, V)와 메트릭(G) 모두 학습합니다.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params (Compressed): {trainable_params:,}")
    
    # 4. 데이터셋 (WikiText 5%)
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir="./results-metric-tuning",
        per_device_train_batch_size=8,
        num_train_epochs=3, # 3 Epoch
        learning_rate=1e-3, # 높은 LR (초기화가 불안정하므로)
        save_strategy="no",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 6. 학습 시작
    print("\n--- Starting Metric Tuning ---")
    start_time = time.time()
    trainer.train()
    print(f"Training Time: {time.time() - start_time:.2f}s")

    # 7. 부활 확인
    print("\n=== Resurrection Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    input_text = "The fundamental laws of physics are"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    print("\nAnalysis:")
    print("- 94%의 뇌세포를 잘라냈음에도 불구하고,")
    print("- 남은 6%의 '핵심 메트릭'만으로 언어 능력을 어느 정도 회복했는지 확인합니다.")

if __name__ == "__main__":
    train_extracted_metric()

