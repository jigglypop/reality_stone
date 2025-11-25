import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from reality_stone.metric_extraction import extract_riemannian_metric
import time

def train_extracted_metric_full_recovery():
    print("=== Reality Stone: Full Recovery Project (6% Brain Challenge) ===")
    
    # 1. 모델 준비
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. 메트릭 추출 (Rank 64, ~94% 압축)
    print("Applying Riemannian Metric Extraction (94% Compressed)...")
    model = extract_riemannian_metric(model, target_dim=64)
    
    # 3. 데이터셋 (WikiText 20% 사용 - 데이터 대폭 증가)
    print("Loading dataset (20% of WikiText)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]")
    
    def tokenize_function(examples):
        # 문맥 길이도 128로 늘려 더 긴 의존성 학습 유도
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. 학습 설정 (본격적인 복구 모드)
    training_args = TrainingArguments(
        output_dir="./results-metric-recovery",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, # 안정적인 그래디언트
        num_train_epochs=5,            # 5 Epoch로 늘림
        learning_rate=5e-4,            # 정교한 튜닝을 위해 LR 약간 낮춤
        warmup_steps=100,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 5. 학습 시작
    print("\n--- Starting Miracle Recovery Training ---")
    start_time = time.time()
    trainer.train()
    print(f"Training Time: {time.time() - start_time:.2f}s")
    
    # 저장 (나중에 써먹기 위해)
    trainer.save_model("./reality-stone-gpt2-compressed")

    # 6. 최종 부활 테스트
    print("\n=== Final Resurrection Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    prompts = [
        "The theory of relativity states that",
        "Artificial intelligence is",
        "The history of the Roman Empire"
    ]
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"\nInput: {p}")
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    train_extracted_metric_full_recovery()

