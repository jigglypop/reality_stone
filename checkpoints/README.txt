# Reality Stone Checkpoints

이 디렉터리는 학습된 모델 체크포인트를 저장합니다.

## 파일 구조

### Joint 모델 (SentenceTopicHead + RCELexicalDecoder)
- `joint_epoch{N}.pt`: N epoch 학습 후 저장된 joint 모델
- `joint_final.pt`: 최종 학습 완료 모델

### Topic Head 단독 모델
- `topic_head_epoch{N}.pt`: N epoch 학습 후 저장된 SentenceTopicHead
- `topic_head_final.pt`: 최종 학습 완료 SentenceTopicHead

### 통합 체크포인트 (HierarchicalSentenceTopicLLM)
- `checkpoint_best.pt`: inference.py에서 사용하는 통합 체크포인트

## 체크포인트 형식

각 체크포인트는 다음 구조를 가집니다:

```python
{
    "config": {
        "d_model": 768,
        "d_head": 64,
        "num_topics": 8,
        "num_heads": 4,
        "vocab_size": 50000,
        "n_layer": 6,
        "n_head": 8,
        "max_length": 128,
    },
    "topic_head": state_dict,  # SentenceTopicHead 파라미터
    "decoder": state_dict,      # RCELexicalDecoder 파라미터
    "epoch": int,
    "loss": float,
}
```

## 사용 방법

### 학습 재개
```bash
python scripts/train.py --checkpoint_dir checkpoints --epochs 20
```

### 추론
```bash
python experiments/inference.py --checkpoint checkpoints/checkpoint_best.pt
```

### QA 테스트
```bash
python experiments/train_qa.py
```

## 정리 가이드

- 오래된 epoch 체크포인트는 주기적으로 삭제 (final만 유지)
- 용량이 큰 경우 Git LFS 사용 권장

