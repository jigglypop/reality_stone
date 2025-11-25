import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from reality_stone.conversion import convert_to_full_riemannian
import math

def train_full_model():
    print("=== Reality Stone: Deep Manifold Fine-Tuning ===")
    
    # 1. 모델 및 토크나이저 로드
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. 리만 수술 집도 (Deep Riemannian Surgery)
    # 곡률 c=1e-4로 설정하여 초기 충격을 완화 (R ~ 100)
    # 모델 내부의 모든 Linear/Conv1D 레이어가 Riemannian Layer로 교체됨
    print("Performing full model surgery...")
    model = convert_to_full_riemannian(model, curvature=1e-4)
    
    # 3. 모델 파라미터 확인
    # 모든 파라미터가 학습 대상이어야 함 (Full Fine-Tuning)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. 데이터셋 준비 (WikiText-2 tiny subset)
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir="./results-full-riemannian",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=5e-5, # Full FT는 낮은 LR 권장
        logging_steps=10,
        save_strategy="no", # 데모용 저장 생략
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 7. 학습 시작
    print("\n--- Starting Training (Re-aligning Manifold Weights) ---")
    trainer.train()
    
    # 8. 학습 후 추론 테스트
    print("\n=== Inference Test after Full Training ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    input_text = "The nature of reality is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"Input: {input_text}")
    print(f"Output: {tokenizer.decode(outputs[0])}")
    print("\nTransformation verified.")

if __name__ == "__main__":
    train_full_model()

