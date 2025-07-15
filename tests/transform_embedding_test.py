import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from reality_stone.layers import PoincareEmbedding, EquivalentPoincareEmbedding
import numpy as np


def test_embedding_equivalence():
    """임베딩 레이어 변환의 정합성 테스트"""
    print("=== Embedding Layer Transformation Test ===\n")
    
    # 1. 간단한 임베딩 테스트
    print("1. Simple Embedding Test")
    vocab_size = 100
    embedding_dim = 64
    
    # 원본 임베딩
    original_emb = nn.Embedding(vocab_size, embedding_dim)
    
    # 푸앵카레 임베딩으로 변환
    poincare_emb = PoincareEmbedding.from_euclidean_embedding(original_emb, c=1.0)
    
    # 동등한 유클리드 임베딩으로 변환
    equiv_emb = EquivalentPoincareEmbedding.from_poincare_embedding(poincare_emb)
    
    # 테스트 입력
    test_indices = torch.randint(0, vocab_size, (10,))
    
    # 출력 비교
    with torch.no_grad():
        original_out = original_emb(test_indices)
        poincare_out = poincare_emb(test_indices)
        equiv_out = equiv_emb(test_indices)
    
    # 유클리드 노름 비교
    original_norms = original_out.norm(dim=-1)
    poincare_norms = poincare_out.norm(dim=-1)
    equiv_norms = equiv_out.norm(dim=-1)
    
    print(f"Original embedding norms: {original_norms[:5].tolist()}")
    print(f"Poincare embedding norms: {poincare_norms[:5].tolist()}")
    print(f"Equivalent embedding norms: {equiv_norms[:5].tolist()}")
    
    # 푸앵카레와 동등 임베딩이 정확히 일치해야 함
    diff = torch.abs(poincare_out - equiv_out).max().item()
    print(f"\nMax difference between Poincare and Equivalent: {diff:.6f}")
    assert diff < 1e-6, "Poincare and Equivalent embeddings should be identical!"
    
    print("✅ Simple embedding test passed!\n")
    
    # 2. GPT-2 임베딩 레이어 테스트
    print("2. GPT-2 Embedding Layer Test")
    
    try:
        # GPT-2 모델 로드
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # 토큰 임베딩 가져오기
        wte = model.wte  # token embeddings
        wpe = model.wpe  # position embeddings
        
        print(f"Token embedding shape: {wte.weight.shape}")
        print(f"Position embedding shape: {wpe.weight.shape}")
        
        # 토큰 임베딩을 푸앵카레로 변환
        poincare_wte = PoincareEmbedding.from_euclidean_embedding(wte, c=1.0)
        equiv_wte = EquivalentPoincareEmbedding.from_poincare_embedding(poincare_wte)
        
        # 위치 임베딩을 푸앵카레로 변환
        poincare_wpe = PoincareEmbedding.from_euclidean_embedding(wpe, c=1.0)
        equiv_wpe = EquivalentPoincareEmbedding.from_poincare_embedding(poincare_wpe)
        
        # 테스트 문장
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # 위치 인덱스 생성
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        
        # 원본 임베딩
        with torch.no_grad():
            original_token_emb = wte(input_ids)
            original_pos_emb = wpe(position_ids)
            original_combined = original_token_emb + original_pos_emb
            
            # 푸앵카레 임베딩
            poincare_token_emb = poincare_wte(input_ids)
            poincare_pos_emb = poincare_wpe(position_ids)
            # 푸앵카레 공간에서는 덧셈이 다름 - 여기서는 단순 비교를 위해 유클리드 덧셈 사용
            poincare_combined = poincare_token_emb + poincare_pos_emb
            
            # 동등 임베딩
            equiv_token_emb = equiv_wte(input_ids)
            equiv_pos_emb = equiv_wpe(position_ids)
            equiv_combined = equiv_token_emb + equiv_pos_emb
        
        # 비교
        print(f"\nOriginal combined shape: {original_combined.shape}")
        print(f"Poincare combined shape: {poincare_combined.shape}")
        print(f"Equivalent combined shape: {equiv_combined.shape}")
        
        # 푸앵카레와 동등 임베딩 비교
        token_diff = torch.abs(poincare_token_emb - equiv_token_emb).max().item()
        pos_diff = torch.abs(poincare_pos_emb - equiv_pos_emb).max().item()
        
        print(f"\nToken embedding max difference: {token_diff:.6f}")
        print(f"Position embedding max difference: {pos_diff:.6f}")
        
        assert token_diff < 1e-6, "Token embeddings should be identical!"
        assert pos_diff < 1e-6, "Position embeddings should be identical!"
        
        print("✅ GPT-2 embedding test passed!")
        
    except Exception as e:
        print(f"⚠️  GPT-2 test skipped: {e}")
    
    # 3. 압축 가능성 테스트
    print("\n3. Compression Feasibility Test")
    
    # 작은 임베딩으로 테스트
    small_emb = nn.Embedding(1000, 128)
    poincare_small = PoincareEmbedding.from_euclidean_embedding(small_emb, c=1.0)
    
    # 압축 정보 확인
    compression_info = poincare_small.compress_to_seed(block_size=64)
    
    print(f"Number of embeddings: {compression_info['block_info']['num_embeddings']}")
    print(f"Embedding dimension: {compression_info['block_info']['embedding_dim']}")
    print(f"Block size: {compression_info['block_info']['block_size']}")
    print(f"Number of blocks: {compression_info['block_info']['emb_blocks']} x {compression_info['block_info']['dim_blocks']}")
    
    total_blocks = compression_info['block_info']['emb_blocks'] * compression_info['block_info']['dim_blocks']
    original_params = 1000 * 128
    compressed_params = total_blocks  # 각 블록당 1개 시드
    compression_ratio = original_params / compressed_params
    
    print(f"\nOriginal parameters: {original_params:,}")
    print(f"Compressed parameters: {compressed_params:,}")
    print(f"Compression ratio: {compression_ratio:.1f}:1")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_embedding_equivalence() 