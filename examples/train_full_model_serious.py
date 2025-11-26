import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from reality_stone.conversion import convert_to_full_riemannian
import math

def train_full_model_serious():
    print("=== Reality Stone: Serious Training ===")
    
    # 1. 모델 및 토크나이저 로드
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. 리만 수술 집도 (곡률을 더 낮추어 충격 최소화)
    print("Performing full model surgery...")
    # curvature를 1e-5로 낮춰 초기 충격을 줄임 (거의 유클리드에 가깝게 시작)
    model = convert_to_full_riemannian(model, curvature=1e-5)
    
    # 3. 데이터셋 준비 (WikiText-2 전체 사용)
    print("Loading FULL dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # 블록 크기를 128로 늘려 문맥 확보
    block_size = 128
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=block_size)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. 학습 설정 (본격적인 Fine-Tuning 파라미터)
    training_args = TrainingArguments(
        output_dir="./results-full-riemannian-serious",
        per_device_train_batch_size=8, # 배치 키움
        gradient_accumulation_steps=4, # 그래디언트 누적
        num_train_epochs=3,            # 3 Epoch
        learning_rate=2e-5,            # 더 섬세한 LR
        weight_decay=0.01,             # 과적합 방지
        warmup_ratio=0.1,              # 웜업
        logging_steps=50,
        save_strategy="epoch",         # 에폭마다 저장
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # 6. 학습 시작
    print("\n--- Starting Serious Training ---")
    trainer.train()
    
    # 7. 저장
    print("Saving trained model...")
    try:
        model.save_pretrained("./reality-stone-phi2-tuned", safe_serialization=False)
        tokenizer.save_pretrained("./reality-stone-phi2-tuned")
    except Exception as e:
        print(f"Save warning: {e}")

    # 8. 추론 테스트
    print("\n=== Inference Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    input_text = "The theory of general relativity suggests that"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"Input: {input_text}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    train_full_model_serious()

