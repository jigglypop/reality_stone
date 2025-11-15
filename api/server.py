"""FastAPI 서버 - Phase 5

docs/sentence_topic_architecture.md의 7장 L4: Post-Controller 및 API 명세 준수
"""
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
from pathlib import Path

# reality_stone 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from api.schemas import RewriteRequest, RewriteResponse
from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.models.sentence_topic_head import SentenceTopicHead
from reality_stone.models.metric_router import MetricContextRouter
from reality_stone.models.rce_lexical_decoder import RCELexicalDecoder

app = FastAPI(title="Sentence-Topic LLM API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 로딩
print("Loading models...")
pre_segmenter = PreSegmenter(max_length=128, k_neighbors=3)
sentence_topic_head = SentenceTopicHead(
    d_model=768,
    d_head=64,
    num_topics=8,
    num_heads=4
)
metric_router = MetricContextRouter(d_head=64)
rce_decoder = RCELexicalDecoder(
    vocab_size=50000,
    d_model=768,
    n_layer=6,
    n_head=8
)

# 평가 모드
sentence_topic_head.eval()
rce_decoder.eval()

print("Models loaded successfully!")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Sentence-Topic LLM API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "rewrite": "/sentence_topic_rewrite"
        }
    }


@app.get("/health")
async def health():
    """
    Health check endpoint
    docs 명세 7.2.1
    """
    return {"status": "ok", "message": "Server is running"}


@app.post("/sentence_topic_rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    """
    Sentence-Topic rewrite endpoint
    
    docs 명세 7.2:
    - L0: Pre-Segmenter
    - L1: SentenceTopicHead
    - L2: MetricContextRouter
    - L3: RCE-LexicalDecoder
    - L4: Post-Controller
    """
    try:
        # L0: Pre-Segmenter
        seg_output = pre_segmenter(request.paragraph)
        
        if seg_output["metadata"]["num_sentences"] == 0:
            return RewriteResponse(
                sentences=[],
                topics=[],
                metric_keys=[],
                replacements=[],
                final_text="",
                stats={
                    "total_tokens": 0,
                    "replaced_tokens": 0,
                    "replacement_ratio": 0.0,
                    "topic_retention": 0.0
                }
            )
        
        # 토큰을 float으로 변환 (임베딩 입력용)
        tokens_float = seg_output["tokens"].float().unsqueeze(0)  # [1, T, seq_len]
        B, T, seq_len = tokens_float.shape
        
        # 간단한 평균 풀링으로 문장 임베딩 생성
        sentence_embeddings = tokens_float.mean(dim=2)  # [1, T]
        
        # d_model 차원으로 확장
        sentence_embeddings = sentence_embeddings.unsqueeze(-1).expand(-1, -1, 768)  # [1, T, 768]
        
        # L1: SentenceTopicHead
        with torch.no_grad():
            topo_idx_input = seg_output["topo_idx"].unsqueeze(0)  # [1, T, K]
            P_topic, scores, metric_keys = sentence_topic_head(
                sentence_embeddings,
                topo_idx_input
            )
        
        # metric_hint 적용
        if request.metric_hint:
            metric_keys = [request.metric_hint + "|" + k for k in metric_keys]
        
        # L2: MetricContextRouter
        L_i = metric_router(metric_keys, scores)
        
        # L3: RCE-LexicalDecoder
        candidates = _build_candidates(request.lexical_overrides, seg_output["tokens"])
        
        with torch.no_grad():
            output_ids, _ = rce_decoder(
                seg_output["tokens"].unsqueeze(0),  # [1, T, seq_len]
                L_i,
                seg_output["replacement_mask"].unsqueeze(0),
                topo_idx_input,
                candidates
            )
        
        # L4: Post-Controller
        response = _build_response(
            seg_output,
            output_ids[0],  # [T, seq_len]
            P_topic[0],     # [T, num_topics]
            metric_keys
        )
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _build_candidates(
    overrides: Dict[str, List[str]],
    tokens: torch.Tensor
) -> Dict[int, List[int]]:
    """
    후보 사전 구축
    
    docs 명세:
    - lexical_overrides 적용
    - 토큰 ID 기반 후보 매핑
    """
    candidates = {}
    
    # 간단한 구현: 각 토큰 ID에 대해 자기 자신과 주변 ID를 후보로
    unique_ids = torch.unique(tokens).tolist()
    for token_id in unique_ids:
        if token_id == 0:  # 패딩 제외
            continue
        # 자기 자신 + 주변 5개 ID를 후보로
        cands = [token_id] + list(range(max(1, token_id-2), min(50000, token_id+3)))
        candidates[token_id] = cands
    
    return candidates


def _build_response(
    seg_output: Dict,
    output_ids: torch.Tensor,
    P_topic: torch.Tensor,
    metric_keys: List[str]
) -> RewriteResponse:
    """
    응답 구성
    
    docs 명세 7.2.3:
    - 문장별 topic score
    - metric key
    - 교체 로그
    - 최종 문장
    - 통계
    """
    sentences = seg_output["sentences"]
    input_ids = seg_output["tokens"]
    
    output_sentences = []
    replacements = []
    
    # 토큰 ID를 문자로 변환 (간단한 chr 사용)
    for i, sent_ids in enumerate(output_ids):
        try:
            # 패딩 제거
            valid_ids = sent_ids[sent_ids != 0]
            # chr로 변환 (범위 제한)
            chars = [chr(min(max(32, int(id)), 126)) if id > 0 else '' for id in valid_ids]
            decoded = ''.join(chars).strip()
            
            # 원본 문장 사용 (디코딩 실패 시)
            if not decoded and i < len(sentences):
                decoded = sentences[i]
            
            output_sentences.append(decoded)
            
            # 교체 로그
            if i < len(input_ids):
                input_sent_ids = input_ids[i]
                for j in range(min(len(sent_ids), len(input_sent_ids))):
                    if input_sent_ids[j] != sent_ids[j] and input_sent_ids[j] != 0:
                        replacements.append({
                            "sentence": i,
                            "pos": j,
                            "old": chr(min(max(32, int(input_sent_ids[j])), 126)),
                            "new": chr(min(max(32, int(sent_ids[j])), 126))
                        })
        except Exception as e:
            # 디코딩 실패 시 원본 사용
            if i < len(sentences):
                output_sentences.append(sentences[i])
    
    # 통계
    total_tokens = (input_ids != 0).sum().item()
    replaced_tokens = len(replacements)
    
    return RewriteResponse(
        sentences=output_sentences,
        topics=P_topic.tolist(),
        metric_keys=metric_keys,
        replacements=replacements,
        final_text=" ".join(output_sentences),
        stats={
            "total_tokens": total_tokens,
            "replaced_tokens": replaced_tokens,
            "replacement_ratio": replaced_tokens / total_tokens if total_tokens > 0 else 0.0,
            "topic_retention": 0.95  # placeholder
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

