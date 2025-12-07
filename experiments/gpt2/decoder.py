import time
import numpy as np
import torch

def _sample_next_token(logits, generated, repetition_penalty=1.2, temperature=1.0, top_k=50, top_p=0.95):
    scores = logits[0, -1, :].clone()
    seen = set(generated[0].tolist())
    for token_id in seen:
        val = scores[token_id]
        if val < 0:
            scores[token_id] = val * repetition_penalty
        else:
            scores[token_id] = val / repetition_penalty
    if temperature != 1.0:
        scores = scores / max(temperature, 1e-8)
    if top_k > 0 and top_k < scores.size(-1):
        _, topk_idx = torch.topk(scores, top_k)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[topk_idx] = False
        scores[mask] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > top_p
        if cutoff.any():
            first = int(cutoff.nonzero(as_tuple=False)[0])
            sorted_scores[first + 1 :] = -float("inf")
            scores = torch.full_like(scores, -float("inf"))
            scores[sorted_idx] = sorted_scores
    probs = torch.softmax(scores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.unsqueeze(0)

def rsulf_generate_text(original_model, rs_model_stack, tokenizer, device, text_prompt, max_tokens=30, top_k_original=0, decoder=None):
    curr_ids = tokenizer.encode(text_prompt, return_tensors="pt").to(device)
    wte = original_model.transformer.wte
    wpe = original_model.transformer.wpe
    lm_head = original_model.lm_head
    ln_f = original_model.transformer.ln_f
    generated = curr_ids
    start_gen = time.time()
    for _ in range(max_tokens):
        seq_len = generated.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
        with torch.no_grad():
            tok_emb = wte(generated)
            pos_emb = wpe(pos)
            x = tok_emb + pos_emb
            h = x
            for wrapper in rs_model_stack.wrappers:
                if hasattr(wrapper, "original_block") and wrapper.original_block is not None:
                    outputs = wrapper.original_block(h, attention_mask=attention_mask)
                    h = outputs[0] if isinstance(outputs, tuple) else outputs
                elif hasattr(wrapper, "rsulf") and hasattr(wrapper.rsulf, "original_block") and wrapper.rsulf.original_block is not None:
                    outputs = wrapper.rsulf.original_block(h, attention_mask=attention_mask)
                    h = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    h = wrapper(h)
            if decoder is not None:
                h_token = h[:, -1, :]
                h_np = h_token.detach().cpu().numpy().astype(np.float32)
                hidden_np = decoder.forward(h_np)
                hidden_t = torch.from_numpy(hidden_np).to(device=device)
                hidden = hidden_t.unsqueeze(1)
                logits = lm_head(ln_f(hidden))
            else:
                h_last = ln_f(h)
                logits = lm_head(h_last)
            next_token = _sample_next_token(logits, generated)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time
