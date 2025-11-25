import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import time
import os

try:
    from reality_stone.metric_extraction import extract_riemannian_metric
except ImportError:
    import sys
    sys.path.append(os.path.join(os.getcwd(), "python"))
    from reality_stone.metric_extraction import extract_riemannian_metric

def fine_tune_extracted_qwen_serious():
    print("=== Reality Stone: Qwen Serious Resurrection (More Epochs & Benchmark) ===")
    
    model_id = "Qwen/Qwen2.5-0.5B"
    save_path = "./data/qwen-extracted"
    tuned_save_path = "./data/qwen-tuned"
    
    # 1. 모델 준비 (재사용)
    if os.path.exists(save_path):
        print(f"Loading extracted model from {save_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(save_path)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
            model = extract_riemannian_metric(model, target_dim=64)
        except Exception as e:
            print(f"로드 실패 ({e}), 새로운 세팅 시작")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
            model = extract_riemannian_metric(model, target_dim=64)
    else:
        print("새로운 모델 준비중.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
        model = extract_riemannian_metric(model, target_dim=64)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 데이터셋 (WikiText 20%)
    print("Loading larger dataset (WikiText 20%)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. 학습 설정 (30 Epochs)
    training_args = TrainingArguments(
        output_dir="./data/qwen-tuning-serious",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=30,  # Increased epochs
        learning_rate=1e-4,
        logging_steps=50,
        save_strategy="epoch",
        fp16=False, # Stability first
        report_to="none",
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("\n--- Starting Serious Training (3 Epochs) ---")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"Training Time: {train_time:.2f}s")
    
    # 저장
    print(f"모델 저장 : {tuned_save_path}...")
    try:
        model.save_pretrained(tuned_save_path, safe_serialization=False)
        tokenizer.save_pretrained(tuned_save_path)
    except Exception as e:
        print(f"저장 실패 : {e}")

    # 4. 벤치마크 (용량 & 속도)
    print("\n=== 벤치마크 결과 ===")
    
    # 용량
    total_size = 0
    if os.path.exists(tuned_save_path):
        for dirpath, _, filenames in os.walk(tuned_save_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    
    size_mb = total_size / (1024*1024)
    # Qwen 0.5B original size is approx 1GB (fp16) or 2GB (fp32)
    # Let's assume 1GB for comparison basis
    orig_size_est = 1000 
    print(f"모델 용량 : {size_mb:.2f} MB")
    print(f"압축비 (Est): {100 * (1 - size_mb/orig_size_est):.2f}% vs 1GB Original")

    # 속도 & 품질
    print("\n--- 추론속도 & 품질 ---")
    model.eval()
    input_text = "딥러닝이란 "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10)
    
    start_inf = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    dur_inf = time.time() - start_inf
    speed = 50 / dur_inf
    
    print(f"Speed: {speed:.2f} tokens/sec")
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    fine_tune_extracted_qwen_serious()

