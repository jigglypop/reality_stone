import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import time


def main():
    parser = argparse.ArgumentParser(description="압축된 모델 Fine-tuning으로 복구")
    parser.add_argument("--model_path", type=str, default="checkpoints/phi2-riemannian")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phi2-riemannian-tuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data_fraction", type=float, default=0.1, help="WikiText 사용 비율")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] 압축된 모델 로드: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")

    print(f"[2/4] WikiText-2 데이터 로드 ({args.data_fraction*100:.0f}%)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    num_samples = int(len(dataset) * args.data_fraction)
    dataset = dataset.select(range(num_samples))
    print(f"   Using {len(dataset)} samples")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"[3/4] Fine-tuning 시작 ({args.epochs} epochs)...")
    start = time.time()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        report_to="none",
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    elapsed = time.time() - start
    print(f"   Training done in {elapsed:.1f}s")

    print(f"[4/4] 저장 및 생성 테스트...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    model.eval()
    prompts = [
        "The theory of relativity states that",
        "인공지능의 미래는",
        "The fundamental laws of physics are"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        print(f"\n[{prompt}]")
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    print(f"\nDone. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()








