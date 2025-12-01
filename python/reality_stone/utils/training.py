import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os
from .misc import get_device

def load_model_and_tokenizer(model_id, cache_dir=None, dtype=torch.float32, device_map=None, local_files_only=False):
    """
    Loads a Causal LM model and tokenizer with standard settings:
    - Sets pad_token = eos_token if missing
    - Handles cache_dir
    - Handles dtype and device_map
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", "E:/hf-cache")
    os.environ.setdefault("HF_HOME", cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir, 
        local_files_only=local_files_only
    )
    
    # Fix padding token
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        device_map=device_map,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True
    )
    
    # Resize embeddings if we added a token
    if hasattr(model, "resize_token_embeddings") and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

def train_model_simple(model, tokenizer, dataset, epochs=3, lr=5e-5, batch_size=4, device=None):
    """
    A simple PyTorch training loop for Causal LM.
    """
    if device is None:
        device = get_device()
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # Handle batch whether it's dict or list
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch.get("labels", input_ids).to(device)
            else:
                # Assume input_ids only
                input_ids = batch.to(device)
                attention_mask = None
                labels = input_ids

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    return model

def generate_text(model, tokenizer, prompts, max_new_tokens=50, temperature=0.7, device=None):
    """
    Generates text for a list of prompts.
    """
    if device is None:
        device = get_device()
        
    model.eval()
    model.to(device)
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append((prompt, generated))
        print(f"\n[{prompt}]")
        print(generated)
        
    return results

