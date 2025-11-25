import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.metric_extraction import extract_riemannian_metric
from reality_stone.hyper_compression import apply_hyper_compression
from tqdm.auto import tqdm
import json


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )
    
    def __len__(self):
        return self.encodings.input_ids.size(0)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx]
        }


def generate_calibration_data(tokenizer, num_samples=32, seq_len=128):
    vocab_size = tokenizer.vocab_size
    random_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    return random_ids.float()


def finetune_compressed_model(model, tokenizer, train_texts, epochs=3, lr=5e-5, batch_size=4, device="cuda"):
    dataset = TextDataset(train_texts, tokenizer)
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
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="gpt2")
    parser.add_argument("--target_dim", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="checkpoints/gpt2-compressed-rs")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "E:/hf-cache"))
    parser.add_argument("--num_steps", type=int, default=100, help="CUDA geometric tuning steps")
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning after compression")
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--data_path", type=str, default=None, help="JSONL file with text data")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"[1/5] Loading model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original params: {orig_params:,}")

    print(f"[2/5] Generating calibration data...")
    calib_data = generate_calibration_data(tok, num_samples=32, seq_len=128)
    print(f"   Calibration shape: {calib_data.shape}")

    print(f"[3/5] Extracting Riemannian Metric (CUDA, {args.num_steps} steps)...")
    model = extract_riemannian_metric(
        model, 
        target_dim=args.target_dim,
        calibration_data=calib_data,
        num_steps=args.num_steps,
        curvature=args.curvature,
        lr=args.lr
    )

    print(f"[4/5] Applying Hyper Compression...")
    model = apply_hyper_compression(model, target_dim=args.target_dim)
    
    new_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - new_params / orig_params) * 100
    print(f"   Compressed params: {new_params:,}")
    print(f"   Reduction: {reduction:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.finetune:
        print(f"[5/5] Fine-tuning compressed model...")
        
        if args.data_path and os.path.exists(args.data_path):
            train_texts = []
            with open(args.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "text" in obj:
                        train_texts.append(obj["text"])
                    elif "paragraph" in obj:
                        train_texts.append(obj["paragraph"])
            print(f"   Loaded {len(train_texts)} samples from {args.data_path}")
        else:
            train_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks can learn complex patterns from data.",
                "Deep learning models require large amounts of training data.",
                "Transformers have revolutionized natural language processing.",
                "Attention mechanisms allow models to focus on relevant parts.",
                "GPT models are trained on massive text corpora.",
                "Language models can generate human-like text.",
            ] * 10
            print(f"   Using {len(train_texts)} synthetic samples")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = finetune_compressed_model(
            model, tok, train_texts,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            device=device
        )
    else:
        print("[5/5] Skipping fine-tuning (use --finetune to enable)")

    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"Saved to {args.save_dir}")

    print("\n[Test] Generating sample text...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    prompt = "The future of AI is"
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.pad_token_id
        )
    
    generated = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()

