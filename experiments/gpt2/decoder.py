import time
import string
import numpy as np
import torch
from reality_stone._rust import PyHumanDecoder

_SKELETON_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "or",
    "if",
    "then",
    "else",
}
_RELATION_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "because",
    "since",
    "therefore",
    "leads",
    "makes",
    "causes",
}

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

def _classify_token(token):
    plain = token.replace("Ġ", "").strip()
    if not plain:
        return "s"
    lower = plain.lower()
    if all(ch in string.punctuation for ch in plain):
        return "s"
    if lower in _SKELETON_WORDS:
        return "s"
    if lower in _RELATION_WORDS or lower.endswith("ing") or lower.endswith("ed"):
        return "r"
    return "o"


def build_human_stage_sets(tokenizer):
    skeleton = []
    relation = []
    objects = []
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        group = _classify_token(token)
        if group == "s":
            skeleton.append(token_id)
        elif group == "r":
            relation.append(token_id)
        else:
            objects.append(token_id)
    if not skeleton and tokenizer.eos_token_id is not None:
        skeleton.append(tokenizer.eos_token_id)
    if not relation and skeleton:
        relation.extend(skeleton[:8])
    if not objects:
        objects.extend(relation or skeleton)
    return skeleton, relation, objects


def build_human_decoder(tokenizer, model, curvature=1e-3):
    embeddings = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)
    skeleton, relation, objects = build_human_stage_sets(tokenizer)
    decoder = PyHumanDecoder(
        embeddings,
        skeleton,
        relation,
        objects,
        alpha_logit=1.0,
        alpha_cos=0.45,
        beta_logit=1.0,
        beta_cos=0.8,
        beta_geo=0.35,
        curvature=curvature,
    )
    return decoder


def _human_decode_token(human_decoder, logits, relation_ctx, object_ctx, device, topk_relation=8, topk_object=24):
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    rel_np = relation_ctx.detach().cpu().numpy().astype(np.float32)
    obj_np = object_ctx.detach().cpu().numpy().astype(np.float32)
    token_ids = human_decoder.decode(logits_np, rel_np, obj_np, topk_relation, topk_object)
    if not token_ids:
        raise RuntimeError("human decoder returned empty selection")
    next_id = int(token_ids[0])
    return torch.tensor([[next_id]], dtype=torch.long, device=device)


def rsulf_generate_text(original_model, rs_model_stack, tokenizer, device, text_prompt, max_tokens=30, top_k_original=0, decoder=None, human_decoder=None, human_cfg=None):
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
            logits_step = logits[:, -1, :]
            use_human = human_decoder is not None
            if use_human:
                relation_ctx = h[:, -1, :]
                object_ctx = h.mean(dim=1)
                cfg = human_cfg or {}
                topk_relation = int(cfg.get("topk_relation", 8))
                topk_object = int(cfg.get("topk_object", 24))
                try:
                    next_token = _human_decode_token(
                        human_decoder,
                        logits_step,
                        relation_ctx,
                        object_ctx,
                        device,
                        topk_relation=topk_relation,
                        topk_object=topk_object,
                    )
                except Exception:
                    next_token = None
            else:
                next_token = None
            if next_token is None:
                next_token = _sample_next_token(
                    logits,
                    generated,
                    repetition_penalty=1.2,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                )
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - start_gen
    return tokenizer.decode(generated[0], skip_special_tokens=True), gen_time
