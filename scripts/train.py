"""학습 스크립트

docs/sentence_topic_implementation.md의 Phase 6 명세 준수
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from reality_stone.models.sentence_topic_head import SentenceTopicHead


class SentenceTopicDataset(Dataset):
    """문장 주제 데이터셋"""
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu'
):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        try:
            tokens = batch["tokens"].to(device)
            topo_idx = batch["topo_idx"].to(device)
            
            # 간단한 평균 풀링으로 문장 임베딩
            B, T, seq_len = tokens.shape
            sentence_embeddings = tokens.float().mean(dim=2).unsqueeze(-1).expand(-1, -1, 768)
            
            # Forward
            P_topic, scores, _ = model(sentence_embeddings, topo_idx)
            
            # 간단한 손실: 균등 분포와의 KL divergence
            uniform_target = torch.ones_like(P_topic) / P_topic.size(-1)
            loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(P_topic + 1e-10),
                uniform_target
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 모델
    model = SentenceTopicHead(
        d_model=args.d_model,
        d_head=args.d_head,
        num_topics=args.num_topics,
        num_heads=args.num_heads
    ).to(device)
    
    # 데이터
    try:
        dataset = SentenceTopicDataset(args.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Windows 호환성
        )
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run: python scripts/prepare_data.py first")
        return
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 학습
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"topic_head_epoch{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # 최종 모델 저장
    final_path = Path(args.checkpoint_dir) / "topic_head_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed_dataset.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--num_topics", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    
    main(args)

