#!/usr/bin/env python3
"""
heegyu/kowikitext-processed-v1 코퍼스를 내려받아
RS-ULF finetune용 텍스트 파일로 변환하는 스크립트.

사용 예시:
    uv run scripts/build_kowikitext_corpus.py \
      --out data/rsulf_finetune_corpus.txt \
      --split train \
      --max_samples 500000
"""

import argparse
from pathlib import Path

from datasets import load_dataset
try:
    from Korpora import Korpora
    HAS_KORPORA = True
except Exception:
    HAS_KORPORA = False


def detect_text_key(example) -> str:
    """
    데이터셋 샘플에서 텍스트 컬럼 이름을 자동으로 추론.
    우선순위: text, content, document, wiki_text, sentence
    """
    candidates = ["text", "content", "document", "wiki_text", "sentence"]
    for key in candidates:
        if key in example and isinstance(example[key], str):
            return key
    # fallback: 첫 번째 str 필드
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise RuntimeError(f"텍스트 컬럼을 찾을 수 없습니다. keys={list(example.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="heegyu/kowikitext-processed-v1",
        help="HuggingFace datasets 이름",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="사용할 split (예: train, validation, test)",
    )
    parser.add_argument(
        "--out",
        default="data/rsulf_finetune_corpus.txt",
        help="출력 텍스트 파일 경로",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="최대 샘플 개수 (-1이면 전체)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"데이터셋 로딩: {args.dataset_name} [{args.split}]")
    print("=" * 70)

    ds = None
    use_hf = False

    # 1차 시도: HuggingFace datasets에서 직접 로드
    try:
        ds = load_dataset(args.dataset_name, split=args.split)
        use_hf = True
        print(f"  HuggingFace datasets로 '{args.dataset_name}' 로드 성공")
    except Exception as e:
        print(f"  {args.dataset_name} 로드 실패: {e}")

    # 2차 시도: 실패하면 Korpora kowikitext로 폴백
    if ds is None:
        if not HAS_KORPORA:
            raise RuntimeError(
                "HuggingFace에서 kowikitext를 로드할 수 없고, "
                "Korpora도 설치되어 있지 않습니다. "
                "uv pip install Korpora 로 설치한 뒤 다시 시도하세요."
            )
        print("  Korpora 'kowikitext'로 폴백합니다.")
        Korpora.fetch("kowikitext")
        corpus = Korpora.load("kowikitext")
        # Korpora kowikitext는 corpus.train에 문장 리스트가 있음
        lines = [getattr(doc, "text", str(doc)) for doc in corpus.train]
    else:
        if len(ds) == 0:
            raise RuntimeError("데이터셋 split이 비어 있습니다.")
        sample = ds[0]
        text_key = detect_text_key(sample)
        print(f"  텍스트 컬럼: {text_key}")
        lines = [ex.get(text_key, "") for ex in ds]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"코퍼스 생성 → {out_path}")
    print("=" * 70)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for text in lines:
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n")
            written += 1
            if args.max_samples > 0 and written >= args.max_samples:
                break

    print(f"  완료: {written} 라인 저장")


if __name__ == "__main__":
    main()


