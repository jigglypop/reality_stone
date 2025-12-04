import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reality_stone.models.transformer_converter import StructuralRSULFModel


class RSULFCausalLM(nn.Module):
    def __init__(self, base_model, rank: int = 128):
        super().__init__()
        self.wte = base_model.transformer.wte
        self.wpe = base_model.transformer.wpe
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = base_model.lm_head
        d_model = base_model.config.n_embd
        blocks = list(base_model.transformer.h)
        self.rsulf_stack = StructuralRSULFModel(blocks, d_model, rank)
        self.rank = rank
        self.d_model = d_model
        self.n_layers = len(blocks)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.rsulf_stack(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def compression_stats(self):
        total_compressed = 0
        total_original = 0
        for layer in self.rsulf_stack.layers:
            c, o = layer.mlp.param_count()
            total_compressed += c
            total_original += o
        ratio = total_original / total_compressed if total_compressed > 0 else 0
        return total_compressed, total_original, ratio


def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    rank = 700
    rsulf_lm = RSULFCausalLM(base_model, rank=rank)
    rsulf_lm.eval()
    compressed, original, ratio = rsulf_lm.compression_stats()
    print(f"[RS-ULF GPT-2] FFN Compression: rank={rank}")
    print(f"  Original FFN params: {original:,}")
    print(f"  Compressed FFN params: {compressed:,}")
    print(f"  Compression ratio: {ratio:.2f}x")
    print()
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generated = input_ids[0].tolist()
    with torch.no_grad():
        for _ in range(30):
            ids = torch.tensor([generated])
            logits = rsulf_lm(ids)
            next_token = int(torch.argmax(logits[0, -1]))
            generated.append(next_token)
            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("[Output]")
    print(text)


if __name__ == "__main__":
    main()


