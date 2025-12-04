import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[Baseline GPT-2]")
    print(text)


if __name__ == "__main__":
    main()


