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
    chunk_size: int = 1000  # 문자 단위
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
            buffer = ""
            paragraph_count = 0
            
            while paragraph_count < max_paragraphs:
                # chunk_size만큼 읽기
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 문단 분리 (빈 줄 기준)
                paragraphs = buffer.split('\n\n')
                
                # 마지막 불완전한 문단은 버퍼에 유지
                buffer = paragraphs[-1]
                paragraphs = paragraphs[:-1]
                
                for para in paragraphs:
                    para = para.strip()
                    if len(para) < 20:  # 너무 짧은 문단 제외
                        continue
                    
                    try:
                        # 전처리
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

