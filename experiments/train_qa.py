import argparse
from pathlib import Path

import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    train_hierarchical_llm_from_text,
    infer_hierarchical_llm_on_text,
    answer_question_from_corpus,
    answer_question_with_llm,
)
from reality_stone.models import HierarchicalLLMConfig


def run_train_qa(
    data: str = "tests/data/text.txt",
    epochs: int = 20,
    batch_size: int = 4,
    max_paragraphs: int = 100,
    device: str = "auto",
    vocab_size: int = 32000,
    lr_backbone: float = 1e-4,
    lr_metric: float = 5e-4,
    checkpoint_dir: str = "checkpoints",
    lambda_consistency_schedule: str = "decay",
    lambda_diversity_schedule: str = "grow",
):
    """
    CLI 플래그 없이 바로 호출할 수 있는 학습 + 데모 + QA 헬퍼.
    (이 파일의 main 로직을 함수 형태로 감싼 버전)
    """
    # 디바이스 자동 선택
    if device == "auto":
        device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_resolved = device

    print(f"Using device: {device_resolved}")

    # 설정
    cfg = HierarchicalLLMConfig(
        vocab_size=vocab_size,
        freeze_decoder=False,
        freeze_topic_head_backbone=False,
        lr_backbone=lr_backbone,
        lr_metric=lr_metric,
        lambda_consistency_schedule=lambda_consistency_schedule,
        lambda_diversity_schedule=lambda_diversity_schedule,
        # 더 작은 모델로 테스트 (빠른 수렴)
        d_model=256,
        d_head=64,
        n_layer_decoder=3,
        n_head_decoder=4,
        max_lm_seq_len=128,
    )

    # 학습
    print("=== 학습 시작 ===")
    model, info = train_hierarchical_llm_from_text(
        data,
        config=cfg,
        epochs=epochs,
        batch_size=batch_size,
        max_paragraphs=max_paragraphs,
        device=device_resolved,
    )

    print("\n=== 학습 완료 ===")
    print(f'Final Loss: {info["final_loss"]:.4f}')
    print(f'Total samples: {info["num_samples"]}')

    # 체크포인트 저장
    checkpoint_path = Path(checkpoint_dir) / "hierarchical_llm_final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "info": info,
        },
        checkpoint_path,
    )
    print(f"\n체크포인트 저장: {checkpoint_path}")

    # 추론 테스트
    print("\n=== 추론 테스트 ===")
    test_samples = [
        "환자는 고혈압 진단을 받았다.",
        "당뇨병 환자의 혈당 수치는",
        "의료진은 신속하게",
    ]

    for sample in test_samples:
        out = infer_hierarchical_llm_on_text(model, sample, max_new_tokens=15)
        print(f"\nINPUT:  {sample}")
        print(f'OUTPUT: {out["generated_text"]}')

    # QA 테스트
    print("\n=== 코퍼스 기반 질의응답(QA) 테스트 ===")
    qa_questions = [
        "이 데이터에서 고혈압 관련 내용은 무엇이야?",
        "치료 계획에 대한 설명을 찾아줘.",
    ]

    for q in qa_questions:
        qa = answer_question_from_corpus(
            model,
            question=q,
            data_path=data,
            max_paragraphs=max_paragraphs,
            top_k=3,
        )
        print(f"\n[QUESTION] {q}")
        for ans in qa["answers"]:
            print(f"  - rank {ans['rank']}: {ans['sentence']}")

        # 계층적 LLM 디코더까지 포함한 직접 응답
        direct = answer_question_with_llm(
            model,
            question=q,
            data_path=data,
            max_paragraphs=max_paragraphs,
            top_k=3,
            max_new_tokens=32,
        )
        print(f"  [LLM answer] {direct['answer']}")

    # 재사용을 위해 반환
    return model, info, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical LLM 학습 및 QA")
    parser.add_argument("--data", type=str, default="tests/data/text.txt", help="학습 데이터 경로")
    parser.add_argument("--epochs", type=int, default=20, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--max_paragraphs", type=int, default=100, help="최대 문단 수")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (auto/cuda/cpu)",
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocab 크기")
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-4,
        help="백본 학습률",
    )
    parser.add_argument(
        "--lr_metric",
        type=float,
        default=5e-4,
        help="메트릭 학습률",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="체크포인트 저장 경로",
    )
    parser.add_argument(
        "--lambda_consistency_schedule",
        type=str,
        default="decay",
        choices=["constant", "decay", "warmup", "grow"],
        help="Consistency loss 스케줄 (decay: 초기 높음→후기 낮음)",
    )
    parser.add_argument(
        "--lambda_diversity_schedule",
        type=str,
        default="grow",
        choices=["constant", "decay", "warmup", "grow"],
        help="Diversity loss 스케줄 (grow: 초기 낮음→후기 높음)",
    )
    args = parser.parse_args()

    run_train_qa(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_paragraphs=args.max_paragraphs,
        device=args.device,
        vocab_size=args.vocab_size,
        lr_backbone=args.lr_backbone,
        lr_metric=args.lr_metric,
        checkpoint_dir=args.checkpoint_dir,
        lambda_consistency_schedule=args.lambda_consistency_schedule,
        lambda_diversity_schedule=args.lambda_diversity_schedule,
    )


if __name__ == "__main__":
    main()


