"""데이터 전처리 스크립트

tests/data/text.txt 데이터셋 전처리
docs/sentence_topic_data_pipeline.md 명세 준수
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reality_stone.utils.pre_segmenter import PreSegmenter
import torch
import json


def prepare_dataset(
    input_file: str = "tests/data/text.txt",
    output_file: str = "data/processed_dataset.pt",
    max_paragraphs: int = 1000,
    max_chars_per_paragraph: int = 4000,
):
    """
    데이터셋 전처리
    
    Args:
        input_file: 입력 텍스트 파일 (200MB+)
        output_file: 출력 파일
        max_paragraphs: 최대 문단 수
        chunk_size: 문단 분할 크기
    """
    print(f"Loading data from {input_file}...")
    
    pre_segmenter = PreSegmenter(max_length=128, k_neighbors=3)

    processed_data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            paragraph_lines = []
            paragraph_count = 0
            current_chars = 0

            for line in f:
                # 빈 줄이면 문단 종료
                if not line.strip():
                    if paragraph_lines:
                        para = " ".join(paragraph_lines).strip()
                        paragraph_lines = []
                        current_chars = 0

                        if len(para) < 20:
                            continue

                        try:
                            seg_output = pre_segmenter(para)

                            if seg_output["metadata"]["num_sentences"] > 0:
                                processed_data.append({
                                    "paragraph": para,
                                    "sentences": seg_output["sentences"],
                                    "tokens": seg_output["tokens"],
                                    "replacement_mask": seg_output["replacement_mask"],
                                    "topo_idx": seg_output["topo_idx"],
                                    "metadata": seg_output["metadata"]
                                })

                                paragraph_count += 1

                                if paragraph_count % 100 == 0:
                                    print(f"Processed {paragraph_count} paragraphs...")

                                if paragraph_count >= max_paragraphs:
                                    break
                        except Exception as e:
                            print(f"Error processing paragraph: {e}")
                            continue
                    continue

                # 빈 줄이 아니면 문단에 추가
                paragraph_lines.append(line.strip())
                current_chars += len(line)

                # 너무 길어지면 강제로 문단 종료 (빈 줄이 없어도)
                if current_chars >= max_chars_per_paragraph:
                    para = " ".join(paragraph_lines).strip()
                    paragraph_lines = []
                    current_chars = 0

                    if len(para) < 20:
                        continue

                    try:
                        seg_output = pre_segmenter(para)

                        if seg_output["metadata"]["num_sentences"] > 0:
                            processed_data.append({
                                "paragraph": para,
                                "sentences": seg_output["sentences"],
                                "tokens": seg_output["tokens"],
                                "replacement_mask": seg_output["replacement_mask"],
                                "topo_idx": seg_output["topo_idx"],
                                "metadata": seg_output["metadata"]
                            })

                            paragraph_count += 1

                            if paragraph_count % 100 == 0:
                                print(f"Processed {paragraph_count} paragraphs...")

                            if paragraph_count >= max_paragraphs:
                                break
                    except Exception as e:
                        print(f"Error processing paragraph: {e}")
                        continue

            # 파일 끝까지 갔을 때 남은 문단 처리
            if paragraph_count < max_paragraphs and paragraph_lines:
                para = " ".join(paragraph_lines).strip()
                if len(para) >= 20:
                    try:
                        seg_output = pre_segmenter(para)
                        if seg_output["metadata"]["num_sentences"] > 0:
                            processed_data.append({
                                "paragraph": para,
                                "sentences": seg_output["sentences"],
                                "tokens": seg_output["tokens"],
                                "replacement_mask": seg_output["replacement_mask"],
                                "topo_idx": seg_output["topo_idx"],
                                "metadata": seg_output["metadata"]
                            })
                    except Exception as e:
                        print(f"Error processing last paragraph: {e}")
                        # 계속 진행

    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return
    
    print(f"\nTotal processed: {len(processed_data)} paragraphs")
    
    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(processed_data, output_file)
    print(f"Saved to {output_file}")
    
    # 통계 출력
    total_sentences = sum(d["metadata"]["num_sentences"] for d in processed_data)
    avg_sentences = total_sentences / len(processed_data) if processed_data else 0
    
    print(f"\nDataset Statistics:")
    print(f"  Total paragraphs: {len(processed_data)}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Avg sentences per paragraph: {avg_sentences:.2f}")


if __name__ == "__main__":
    prepare_dataset()

