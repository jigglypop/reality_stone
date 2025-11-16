"""학습 스크립트

tests/data/text.txt 원본 데이터를 직접 사용하여
SentenceTopicHead를 비지도로 학습한다.

docs/sentence_topic_implementation.md의 Phase 6 명세를
전처리 파일 없이 최소한으로 만족하는 형태로 구현.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from reality_stone.models.sentence_topic_head import SentenceTopicHead
from reality_stone.models.metric_router import MetricContextRouter
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder
from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.data import SentenceTopicDataset, collate_batch


# 하위 호환성을 위해 여기서도 Dataset 클래스를 정의하지만, 실제로는 reality_stone.data를 사용
# (이 파일의 SentenceTopicDataset은 deprecated)
class _LegacySentenceTopicDataset(Dataset):
    """
    문장 주제 데이터셋.

    - 입력: 원본 텍스트 파일 (예: tests/data/text.txt)
    - 전처리 파일(.pt)을 따로 생성하지 않고, 학습 스크립트 내부에서
      문단 단위로 PreSegmenter를 적용해 바로 사용한다.
    """

    def __init__(
        self,
        data_path: str,
        max_paragraphs: int = 1000,
        min_chars: int = 20,
        max_chars_per_paragraph: int = 4000,
    ):
        self.samples = []
        self._build_from_raw_text(
            data_path,
            max_paragraphs=max_paragraphs,
            min_chars=min_chars,
            max_chars_per_paragraph=max_chars_per_paragraph,
        )

    def _build_from_raw_text(
        self,
        data_path: str,
        max_paragraphs: int,
        min_chars: int,
        max_chars_per_paragraph: int,
    ) -> None:
        pre_segmenter = PreSegmenter(max_length=128, k_neighbors=3)
        paragraph_lines = []
        paragraph_count = 0
        current_chars = 0

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        # 문단 종료
                        if paragraph_lines:
                            para = " ".join(paragraph_lines).strip()
                            paragraph_lines = []
                            current_chars = 0

                            if len(para) >= min_chars:
                                try:
                                    seg_output = pre_segmenter(para)
                                    if seg_output["metadata"]["num_sentences"] > 0:
                                        self.samples.append(
                                            {
                                                "paragraph": para,
                                                "sentences": seg_output["sentences"],
                                                "tokens": seg_output["tokens"],
                                                "replacement_mask": seg_output[
                                                    "replacement_mask"
                                                ],
                                                "topo_idx": seg_output["topo_idx"],
                                                "metadata": seg_output["metadata"],
                                            }
                                        )
                                        paragraph_count += 1

                                        if paragraph_count % 100 == 0:
                                            print(
                                                f"[Dataset] Loaded {paragraph_count} paragraphs..."
                                            )

                                        if paragraph_count >= max_paragraphs:
                                            break
                                except Exception as e:
                                    print(f"[Dataset] Error processing paragraph: {e}")
                                    continue
                        continue

                    # 빈 줄이 아니면 현재 문단에 추가
                    paragraph_lines.append(line.strip())
                    current_chars += len(line)

                    if current_chars >= max_chars_per_paragraph:
                        para = " ".join(paragraph_lines).strip()
                        paragraph_lines = []
                        current_chars = 0

                        if len(para) < min_chars:
                            continue

                        try:
                            seg_output = pre_segmenter(para)
                            if seg_output["metadata"]["num_sentences"] > 0:
                                self.samples.append(
                                    {
                                        "paragraph": para,
                                        "sentences": seg_output["sentences"],
                                        "tokens": seg_output["tokens"],
                                        "replacement_mask": seg_output[
                                            "replacement_mask"
                                        ],
                                        "topo_idx": seg_output["topo_idx"],
                                        "metadata": seg_output["metadata"],
                                    }
                                )
                                paragraph_count += 1

                                if paragraph_count % 100 == 0:
                                    print(
                                        f"[Dataset] Loaded {paragraph_count} paragraphs..."
                                    )

                                if paragraph_count >= max_paragraphs:
                                    break
                        except Exception as e:
                            print(f"[Dataset] Error processing paragraph: {e}")
                            continue

                # 파일 끝까지 읽은 뒤 남은 문단 처리
                if paragraph_count < max_paragraphs and paragraph_lines:
                    para = " ".join(paragraph_lines).strip()
                    if len(para) >= min_chars:
                        try:
                            seg_output = pre_segmenter(para)
                            if seg_output["metadata"]["num_sentences"] > 0:
                                self.samples.append(
                                    {
                                        "paragraph": para,
                                        "sentences": seg_output["sentences"],
                                        "tokens": seg_output["tokens"],
                                        "replacement_mask": seg_output[
                                            "replacement_mask"
                                        ],
                                        "topo_idx": seg_output["topo_idx"],
                                        "metadata": seg_output["metadata"],
                                    }
                                )
                        except Exception as e:
                            print(f"[Dataset] Error processing last paragraph: {e}")
        except FileNotFoundError:
            print(f"[Dataset] Error: {data_path} not found")

        print(f"[Dataset] Total loaded paragraphs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# collate_batch는 reality_stone.data에서 import됨 (상단 참조)
def _legacy_collate_batch(batch):
    """
    가변 길이 문단(T, seq_len)이 섞인 배치를 하나의 텐서로 패딩하여 합친다.

    - tokens: [B, T_max, L_max]
    - replacement_mask: [B, T_max, L_max]
    - topo_idx: [B, T_max, K_max]
      * pad된 위치의 이웃 인덱스는 마지막 문장 인덱스(or 0)로 채워서
        SentenceTopicHead / MetricAttention에서 유효한 인덱스로 유지한다.
    """
    import torch

    max_T = max(item["tokens"].shape[0] for item in batch)
    max_L = max(item["tokens"].shape[1] for item in batch)
    max_K = max(item["topo_idx"].shape[1] for item in batch)

    tokens_batch = []
    masks_batch = []
    topo_batch = []

    for item in batch:
        tokens = item["tokens"]
        mask = item["replacement_mask"]
        topo = item["topo_idx"]
        T, L = tokens.shape
        K = topo.shape[1]

        # tokens 패딩: 0으로 채움
        tokens_padded = torch.zeros((max_T, max_L), dtype=tokens.dtype)
        tokens_padded[:T, :L] = tokens
        tokens_batch.append(tokens_padded)

        # replacement_mask 패딩: 0으로 채움 (0 = 고정, 1 = 교체 가능)
        mask_padded = torch.zeros((max_T, max_L), dtype=mask.dtype)
        mask_padded[:T, :L] = mask
        masks_batch.append(mask_padded)

        # topo_idx 패딩
        topo_padded = torch.empty((max_T, max_K), dtype=topo.dtype)
        # 기존 부분 복사
        topo_padded[:T, :K] = topo

        # 열(K) 방향 패딩: 마지막 이웃 인덱스를 반복
        if K < max_K:
            last_col = topo[:, -1:]
            topo_padded[:T, K:] = last_col.expand(T, max_K - K)

        # 행(T) 방향 패딩: 마지막 문장 인덱스(또는 0)를 사용
        fill_idx = max(0, T - 1)
        if T < max_T:
            topo_padded[T:, :] = fill_idx

        topo_batch.append(topo_padded)

    return {
        "tokens": torch.stack(tokens_batch, dim=0),
        "replacement_mask": torch.stack(masks_batch, dim=0),
        "topo_idx": torch.stack(topo_batch, dim=0),
    }


def train_epoch(
    topic_head: nn.Module,
    metric_router: MetricContextRouter,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    lambda_consistency: float = 1.0,
    lambda_diversity: float = 0.1,
):
    """
    1 epoch joint 학습 (완전 비지도, 문단 단위):

    - SentenceTopicHead + RCELexicalDecoder 를 함께 학습
    - 문단 내 문장들은 비슷한 topic 분포를 갖도록 (consistency)
    - 전체 배치에서는 다양한 topic이 사용되도록 (diversity)
    - 디코더는 첫 토큰을 자기 자신으로 재구성하도록 autoencoding (lexical constraint 포함)
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    topic_head.train()
    decoder.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar, start=1):
        try:
            tokens = batch["tokens"].to(device)              # [B, T, seq_len]
            topo_idx = batch["topo_idx"].to(device)          # [B, T, K]
            repl_mask = batch["replacement_mask"].to(device) # [B, T, seq_len]

            B, T, seq_len = tokens.shape

            # 문장 임베딩: 토큰 평균 → [B,T,d_model]
            sentence_embeddings = tokens.float().mean(dim=2)              # [B, T]
            sentence_embeddings = sentence_embeddings.unsqueeze(-1).expand(-1, -1, 768)

            # L1: SentenceTopicHead
            P_topic, scores, metric_keys = topic_head(sentence_embeddings, topo_idx)
            C = P_topic.size(-1)

            # 1) 문단 내 consistency
            paragraph_mean = P_topic.mean(dim=1, keepdim=True).detach()   # [B,1,C]
            paragraph_mean = paragraph_mean.expand(-1, T, -1)             # [B,T,C]
            log_p = (P_topic + 1e-10).log()
            loss_consistency = nn.KLDivLoss(reduction="batchmean")(log_p, paragraph_mean)

            # 2) 전체 diversity
            batch_mean = P_topic.mean(dim=(0, 1))  # [C]
            uniform = torch.full_like(batch_mean, 1.0 / C)
            loss_diversity = nn.KLDivLoss(reduction="batchmean")(
                (batch_mean + 1e-10).log(), uniform
            )

            # L2: MetricContextRouter
            # scores는 GPU에 있으나, router는 CPU 텐서를 사용하는 경향이 있어 detach 후 넘긴다.
            L_i = metric_router(metric_keys, scores.detach().cpu())

            # L3: RCELexicalDecoder
            # 문장별 첫 번째 토큰만 사용 (inference.py와 동일한 방식)
            tokens_input = tokens[:, :, 0].clamp(0, decoder.vocab_size - 1)      # [B, T]
            mask_input = repl_mask[:, :, 0]                                      # [B, T]

            # candidates: 원본 토큰 ID를 반드시 포함하는 작은 후보 집합
            unique_tokens = torch.unique(tokens_input)
            candidates = {
                int(tid): [
                    int(tid),
                    min(int(tid) + 1, decoder.vocab_size - 1),
                    min(int(tid) + 2, decoder.vocab_size - 1),
                ]
                for tid in unique_tokens.tolist()
                if tid > 0
            }

            # 디코더 forward
            output_ids, logits = decoder(tokens_input, L_i, mask_input, topo_idx, candidates)

            # 3) Autoencoding loss: 첫 토큰을 자기 자신으로 복원하도록 CrossEntropy
            #    (lexical constraint는 디코더 내부에서 이미 적용됨)
            V = logits.size(-1)
            logits_flat = logits.view(-1, V)
            targets_flat = tokens_input.view(-1)
            mask_flat = mask_input.view(-1) > 0  # 교체 가능 위치에만 손실 적용

            if mask_flat.any():
                ce_loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
            else:
                ce_loss = torch.tensor(0.0, device=device)

            loss = (
                ce_loss
                + lambda_consistency * loss_consistency
                + lambda_diversity * loss_diversity
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(topic_head.parameters()) + list(decoder.parameters()),
                1.0,
            )
            optimizer.step()

            total_loss += float(loss.item())

            # tqdm 프로그레스 바에 현재 loss를 표시 (10 step마다 업데이트)
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{ce_loss.item():.4f}",
                    cons=f"{loss_consistency.item():.4f}",
                    div=f"{loss_diversity.item():.4f}",
                )
        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def run_eval_demo(
    topic_head: nn.Module,
    metric_router: MetricContextRouter,
    decoder: nn.Module,
    device: str,
    text: str,
    prefix: str = "",
):
    """
    현재 joint 모델 상태로 간단한 추론을 수행하고,
    사람 읽을 수 있는 형태로 원문/재작성 문장을 출력한다.
    """
    from reality_stone.utils.pre_segmenter import PreSegmenter
    import torch

    topic_head.eval()
    decoder.eval()

    segmenter = PreSegmenter(max_length=128, k_neighbors=3)
    print(f"\n[{prefix}Demo] input: {text[:80]}...")

    seg_output = segmenter(text)
    if seg_output["metadata"]["num_sentences"] == 0:
        print(f"[{prefix}Demo] No sentences found.")
        return

    tokens = seg_output["tokens"].unsqueeze(0).to(device)  # [1, T, L]
    topo_idx = seg_output["topo_idx"].unsqueeze(0).to(device)
    repl_mask = seg_output["replacement_mask"].unsqueeze(0).to(device)

    # 문장 임베딩
    sent_emb = tokens.float().mean(dim=2)
    sent_emb = sent_emb.unsqueeze(-1).expand(-1, -1, topic_head.d_model)

    with torch.no_grad():
        P_topic, scores, metric_keys = topic_head(sent_emb, topo_idx)
        L_i = metric_router(metric_keys, scores.detach().cpu())

        tokens_input = tokens[:, :, 0].clamp(0, decoder.vocab_size - 1)  # [1, T]
        mask_input = repl_mask[:, :, 0]                                  # [1, T]

        unique_tokens = torch.unique(tokens_input)
        candidates = {
            int(tid): [
                int(tid),
                min(int(tid) + 1, decoder.vocab_size - 1),
                min(int(tid) + 2, decoder.vocab_size - 1),
            ]
            for tid in unique_tokens.tolist()
            if tid > 0
        }

        output_ids, _ = decoder(tokens_input, L_i, mask_input, topo_idx, candidates)

    original_sentences = seg_output["sentences"]
    token_matrix = seg_output["tokens"]  # [T, L]
    tokenizer = segmenter.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0

    rewritten_sentences = []
    for idx, sent in enumerate(original_sentences):
        if tokenizer is None:
            rewritten_sentences.append(sent)
            continue

        orig_ids = token_matrix[idx].tolist()
        new_ids = list(orig_ids)
        new_token_id = int(output_ids[0, idx].cpu().item())
        if new_token_id != pad_id:
            new_ids[0] = new_token_id

        trimmed_ids = [tid for tid in new_ids if tid != pad_id]
        try:
            decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        except Exception:
            decoded = sent
        rewritten_sentences.append(decoded)

    rewritten_text = " ".join(rewritten_sentences)

    print(f"\n[{prefix}Demo] Original / Rewritten sentences:")
    for i, (orig, rew) in enumerate(zip(original_sentences, rewritten_sentences), 1):
        print(f"  [{i}] {orig} -> {rew}")
    print(f"\n[{prefix}Demo] Rewritten paragraph: {rewritten_text}")


def main(args):
    # GPU가 있으면 기본적으로 사용 (사용자가 굳이 끄지 않는 한)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 빠른 실험용 설정(--quick)이면 에폭/문단 수 줄이기
    if getattr(args, "quick", False):
        # 이미 지정된 값들보다 더 줄이지는 않도록 min 사용
        args.epochs = min(args.epochs, 2)
        args.max_paragraphs = min(args.max_paragraphs, 200)
        args.batch_size = min(args.batch_size, 4)
        print(
            f"[Quick mode] epochs={args.epochs}, "
            f"max_paragraphs={args.max_paragraphs}, batch_size={args.batch_size}"
        )
    
    # 모델 구성요소 (joint 학습)
    topic_head = SentenceTopicHead(
        d_model=args.d_model,
        d_head=args.d_head,
        num_topics=args.num_topics,
        num_heads=args.num_heads,
    ).to(device)
    metric_router = MetricContextRouter(d_head=args.d_head)
    decoder = RCELexicalDecoder(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.num_heads,
    ).to(device)
    
    # 데이터
    try:
        dataset = SentenceTopicDataset(
            args.data_path,
            max_paragraphs=args.max_paragraphs,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Windows 호환성
            collate_fn=collate_batch,
        )
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run: python scripts/prepare_data.py first")
        return
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(topic_head.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )
    
    # 학습
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(
            topic_head,
            metric_router,
            decoder,
            dataloader,
            optimizer,
            device,
            lambda_consistency=args.lambda_consistency,
            lambda_diversity=args.lambda_diversity,
        )
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

        # 에폭 단위 joint LLM 데모 (질문/문단에 대한 응답 확인)
        if getattr(args, "eval_text", None):
            run_eval_demo(
                topic_head,
                metric_router,
                decoder,
                device,
                args.eval_text,
                prefix=f"epoch{epoch+1} ",
            )
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"joint_epoch{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "config": {
                    "d_model": args.d_model,
                    "d_head": args.d_head,
                    "num_topics": args.num_topics,
                    "num_heads": args.num_heads,
                    "vocab_size": args.vocab_size,
                    "n_layer": args.n_layer,
                    "n_head": args.num_heads,
                    "max_length": 128,
                },
                "topic_head": topic_head.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "loss": loss,
            }
            torch.save(ckpt, checkpoint_path)
            print(f"Saved joint checkpoint to {checkpoint_path}")
    
    # 최종 모델 저장
    final_path = Path(args.checkpoint_dir) / "joint_final.pt"
    final_ckpt = {
        "config": {
            "d_model": args.d_model,
            "d_head": args.d_head,
            "num_topics": args.num_topics,
            "num_heads": args.num_heads,
            "vocab_size": args.vocab_size,
            "n_layer": args.n_layer,
            "n_head": args.num_heads,
            "max_length": 128,
        },
        "topic_head": topic_head.state_dict(),
        "decoder": decoder.state_dict(),
        "epoch": args.epochs,
        "loss": loss,
    }
    torch.save(final_ckpt, final_path)
    print(f"\nTraining complete! Final joint model saved to {final_path}")

    # 학습 종료 후 최종 joint LLM 데모 (사람이 읽을 수 있는 문장 재구성)
    if getattr(args, "demo_text", None):
        run_eval_demo(
            topic_head,
            metric_router,
            decoder,
            device,
            args.demo_text,
            prefix="final ",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="tests/data/text.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--num_topics", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--max_paragraphs", type=int, default=1000)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_diversity", type=float, default=0.1)
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="(deprecated) GPU 사용 플래그 - 현재는 GPU가 있으면 자동 사용",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 실험용 설정 (에폭/문단 수/배치 크기 자동 축소)",
    )
    parser.add_argument(
        "--demo_text",
        type=str,
        default=None,
        help="학습 완료 후 joint LLM으로 데모 응답을 보고 싶을 때 사용하는 입력 텍스트",
    )
    parser.add_argument(
        "--eval_text",
        type=str,
        default=None,
        help="각 epoch 끝마다 joint LLM으로 추론해볼 평가용 텍스트(질문/문단)",
    )
    args = parser.parse_args()
    
    main(args)

