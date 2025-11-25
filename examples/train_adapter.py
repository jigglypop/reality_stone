import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from reality_stone.adapter import patch_llm_with_reality_stone
import os

def train_adapter():
    print("=== Reality Stone Adapter Training Demo ===")
    
    # 1. 모델 준비
    model_id = "gpt2" # 가벼운 모델로 시작
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. Reality Stone 패치 적용
    # init_scale을 조정하여 기존 임베딩 크기와 유사하게 맞춤
    model = patch_llm_with_reality_stone(model, diffusion_steps=2, alpha=0.5)
    
    # 3. 기존 파라미터 동결 (Freezing)
    # 오직 Reality Stone 어댑터(Diffusion Head, Hyperbolic Scale)만 학습
    for name, param in model.named_parameters():
        if "diffusion_engine" in name or "flow_net" in name or "output_scale" in name:
            param.requires_grad = True
            print(f"Training: {name}")
        else:
            param.requires_grad = False
            
    # 4. 데이터셋 준비 (예: wikitext-2-raw-v1 small sample)
    # 실제로는 'reality_stone/data'의 로더를 사용하거나 커스텀 데이터를 사용
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]") # 빠른 데모를 위해 1%만
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir="./results-reality-stone",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-3, # 어댑터는 약간 높은 LR 가능
        logging_steps=10,
        save_strategy="no", # Disable auto-saving to avoid weight tying issues during demo
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 6. Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving adapter model...")
    try:
        model.save_pretrained("./reality-stone-gpt2", safe_serialization=False)
    except Exception as e:
        print(f"Warning: Could not save model due to weight tying issues: {e}")
        print("Proceeding to inference test...")
    
    # 7. 추론 테스트
    print("\n=== Inference Test after Training ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer("The concept of curved space implies", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    print(f"Output: {tokenizer.decode(outputs[0])}")

if __name__ == "__main__":
    train_adapter()

