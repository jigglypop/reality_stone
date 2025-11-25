import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.conversion import convert_to_full_riemannian
from reality_stone.metric_extraction import extract_riemannian_metric
from tqdm.auto import tqdm
import json


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt"
        )
    
    def __len__(self):
        return self.encodings.input_ids.size(0)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx]
        }


def train_model(model, tokenizer, texts, epochs=3, lr=5e-5, batch_size=2, device="cuda"):
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--target_dim", type=int, default=32)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rs-pipeline")
    parser.add_argument("--cache_dir", type=str, default="E:/hf-cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/5] 모델 로드: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, cache_dir=args.cache_dir
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original: {orig_params:,} params")

    print(f"[2/5] 리만 변환 (curvature={args.curvature})...")
    model = convert_to_full_riemannian(model, curvature=args.curvature)

    print(f"[3/5] 메트릭 추출 (dim={args.target_dim})...")
    model = extract_riemannian_metric(
        model, target_dim=args.target_dim,
        calibration_data=None, num_steps=1,
        curvature=args.curvature, lr=0.0
    )

    new_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - new_params / orig_params) * 100
    print(f"   Compressed: {new_params:,} params ({reduction:.1f}% reduction)")

    print(f"[4/5] Fine-tuning ({args.epochs} epochs) with WikiText...")
    from datasets import load_dataset
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    num_samples = min(5000, len(dataset))
    texts = [t for t in dataset["text"][:num_samples] if len(t.strip()) > 50]
    print(f"   Using {len(texts)} WikiText samples")

    model = train_model(model, tokenizer, texts, epochs=args.epochs, batch_size=4, device=device)

    print(f"[5/5] 생성 테스트...")
    model.eval()
    
    prompts = [
        "The theory of relativity",
        "인공지능의 미래는",
        "Deep learning"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=40,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        print(f"\n[{prompt}]")
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.save_dir}/model.pt")
    tokenizer.save_pretrained(args.save_dir)
    print(f"\nDone. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()

