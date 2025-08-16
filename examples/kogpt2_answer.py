import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='skt/kogpt2-base-v2')
	parser.add_argument('--prompt', default='''아래는 사용자와 도우미의 대화이다. 도우미는 한국어로 차분하고 상세하게 설명한다.

사용자: 하이퍼볼릭 신경망과 Poincaré 모델을 간단히 설명하고, 현실 시스템(RAG)에서 왜 곡률을 조절하는 전처리가 유용한지 설명해 줘. 예시는 간단한 수식과 함께.
도우미:''')
	parser.add_argument('--max-new-tokens', type=int, default=180)
	parser.add_argument('--device', default='cpu')
	parser.add_argument('--temperature', type=float, default=0.8)
	parser.add_argument('--top-p', type=float, default=0.9)
	args = parser.parse_args()

	tok = AutoTokenizer.from_pretrained(args.model)
	model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()

	input_ids = tok.encode(args.prompt, return_tensors='pt').to(model.device)
	# warmup
	with torch.inference_mode():
		_ = model.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)

	start = time.time()
	with torch.inference_mode():
		out = model.generate(
			input_ids,
			max_new_tokens=args.max_new_tokens,
			do_sample=True,
			temperature=args.temperature,
			top_p=args.top_p,
			pad_token_id=tok.eos_token_id,
		)
	latency = time.time() - start
	gen_tokens = out.shape[-1] - input_ids.shape[-1]
	tokens_per_sec = gen_tokens / max(1e-6, latency)

	text = tok.decode(out[0], skip_special_tokens=True)
	print(text)
	print(f"\n[metrics] latency_sec={latency:.3f}, gen_tokens={gen_tokens}, tokens_per_sec={tokens_per_sec:.2f}")


if __name__ == '__main__':
	main()
