"""계층적 문장-주제 LLM 학습 스크립트

이 스크립트는 SentenceTopicHead와 HierarchicalLMDecoder 구조를 사용하여
문단(Paragraph) -> 주제(Topic) -> 문장(Sentence) 계층 구조를 학습합니다.
"""
import sys
from pathlib import Path

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import torch
import argparse
from reality_stone.models.hierarchical_sentence_topic_llm import (
    train_hierarchical_llm_from_text,
    HierarchicalLLMConfig,
)

def main(args):
    from reality_stone.utils.misc import get_device
    device = get_device()
    print(f"사용 디바이스: {device}")

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.max_paragraphs = min(args.max_paragraphs, 200)
        args.batch_size = min(args.batch_size, 4)
        print(f"[빠른 학습 모드] epochs={args.epochs}, max_paragraphs={args.max_paragraphs}")

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
        max_lm_seq_len=512,
    )

    print(f"학습 시작 (설정: {config})")

    teacher_model = None
    teacher_tokenizer = None
    kd_proj = None

    # 지식 증류 (Knowledge Distillation) 설정
    if args.teacher_model and args.kd_weight > 0.0:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[KD] 교사 모델 로딩 중: {args.teacher_model}")
            teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
            
            # 패딩 토큰 설정
            if getattr(teacher_tokenizer, "pad_token", None) is None:
                if getattr(teacher_tokenizer, "eos_token", None) is not None:
                    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
                else:
                    teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            
            # 임베딩 크기 조정
            if hasattr(teacher_tokenizer, "vocab") and hasattr(teacher_model, "resize_token_embeddings"):
                try:
                    teacher_model.resize_token_embeddings(len(teacher_tokenizer))
                except Exception:
                    pass
            
            teacher_model.eval()
            hidden_size = getattr(teacher_model.config, "hidden_size", config.d_model)
            kd_proj = torch.nn.Linear(hidden_size, config.d_model).to(device)
            
        except Exception as e:
            print(f"[KD] 비활성화 (교사 모델 로드 오류): {e}")
            teacher_model = None
            teacher_tokenizer = None
            kd_proj = None

    # 모델 학습 실행
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
    print(f"\n학습 완료! 최종 모델 저장됨: {checkpoint_path}")
    
    # 데모 추론 (선택 사항)
    if args.demo_text:
        from reality_stone.models.hierarchical_sentence_topic_llm import infer_hierarchical_llm_on_text
        print(f"\n[데모] 입력: {args.demo_text}")
        result = infer_hierarchical_llm_on_text(
            model=model,
            text=args.demo_text,
            max_new_tokens=20
        )
        print(f"[데모] 생성 결과: {result['generated_text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="tests/data/text.txt", help="학습 데이터 경로")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="체크포인트 저장 경로")
    parser.add_argument("--d_model", type=int, default=768, help="모델 차원 크기")
    parser.add_argument("--d_head", type=int, default=64, help="헤드 차원 크기")
    parser.add_argument("--num_topics", type=int, default=8, help="주제(Topic) 개수")
    parser.add_argument("--num_heads", type=int, default=4, help="어텐션 헤드 개수")
    parser.add_argument("--n_layer", type=int, default=6, help="디코더 레이어 수")
    parser.add_argument("--vocab_size", type=int, default=32000, help="어휘 사전 크기 (BERT 기준)")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--max_paragraphs", type=int, default=1000, help="최대 학습 문단 수")
    parser.add_argument("--lambda_consistency", type=float, default=1.0, help="일관성 손실 가중치")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="다양성 손실 가중치")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트 모드")
    parser.add_argument("--demo_text", type=str, default=None, help="학습 후 테스트할 데모 텍스트")
    parser.add_argument("--teacher_model", type=str, default=None, help="지식 증류용 교사 모델 ID")
    parser.add_argument("--kd_weight", type=float, default=0.0, help="지식 증류 손실 가중치")
    
    args = parser.parse_args()
    main(args)
