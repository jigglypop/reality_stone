"""학습 스크립트
SentenceTopicHead와 HierarchicalLMDecoder를 사용하여 
Sentence-Topic LLM을 학습한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
import argparse
from reality_stone.models.hierarchical_sentence_topic_llm import (
    train_hierarchical_llm_from_text,
    HierarchicalLLMConfig,
)

def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.max_paragraphs = min(args.max_paragraphs, 200)
        args.batch_size = min(args.batch_size, 4)
        print(f"[Quick mode] epochs={args.epochs}, max_paragraphs={args.max_paragraphs}")

    config = HierarchicalLLMConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_head=args.d_head,
        num_topics=args.num_topics,
        num_heads_topic=args.num_heads,
        n_layer_decoder=args.n_layer,
        n_head_decoder=args.num_heads,
        lambda_consistency=args.lambda_consistency,
        lambda_diversity=args.lambda_diversity,
        use_pretrained_embeddings=True,
        pretrained_tokenizer="klue/bert-base",
    )

    print(f"Starting training with config: {config}")

    teacher_model = None
    teacher_tokenizer = None
    kd_proj = None

    if args.teacher_model and args.kd_weight > 0.0:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            print(f"[KD] Loading teacher model: {args.teacher_model}")
            teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            teacher_model.eval()
            hidden_size = getattr(teacher_model.config, "hidden_size", config.d_model)
            kd_proj = torch.nn.Linear(hidden_size, config.d_model).to(device)
        except Exception as e:
            print(f"[KD] Disabled (error loading teacher): {e}")
            teacher_model = None
            teacher_tokenizer = None
            kd_proj = None

    model, info = train_hierarchical_llm_from_text(
        data_path=args.data_path,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_paragraphs=args.max_paragraphs,
        device=device,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        kd_proj=kd_proj,
        kd_weight=args.kd_weight,
    )
    
    checkpoint_path = Path(args.checkpoint_dir) / "joint_final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "d_head": config.d_head,
        "num_topics": config.num_topics,
        "num_heads": config.num_heads_topic,
        "n_layer": config.n_layer_decoder,
        "n_head": config.n_head_decoder,
        "use_pretrained_embeddings": config.use_pretrained_embeddings,
    }

    ckpt = {
        "config": config_dict,
        "topic_head": model.topic_head.state_dict(),
        "decoder": model.decoder.state_dict(),
        "epoch": args.epochs,
        "loss": info.get("final_loss", 0.0),
    }

    torch.save(ckpt, checkpoint_path)
    print(f"\nTraining complete! Final model saved to {checkpoint_path}")
    
    # Demo Inference if requested
    if args.demo_text:
        from reality_stone.models.hierarchical_sentence_topic_llm import infer_hierarchical_llm_on_text
        print(f"\n[Demo] Input: {args.demo_text}")
        result = infer_hierarchical_llm_on_text(
            model=model,
            text=args.demo_text,
            max_new_tokens=20
        )
        print(f"[Demo] Generated: {result['generated_text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="tests/data/text.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--num_topics", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=32000) # BERT vocab size
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_paragraphs", type=int, default=1000)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_diversity", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--demo_text", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--kd_weight", type=float, default=0.0)
    
    args = parser.parse_args()
    main(args)
