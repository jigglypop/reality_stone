"""문단 분해 및 전처리 모듈 - Phase 1"""
import torch
from typing import List, Dict, Tuple
import re

class PreSegmenter:
    """
    문단을 문장 단위로 분해하고 토큰화, topology index, replacement mask 생성
    
    docs/sentence_topic_architecture.md의 3장 L0: Pre-Segmenter 명세 준수
    """
    def __init__(
        self,
        max_length: int = 128,
        k_neighbors: int = 3
    ):
        self.max_length = max_length
        self.k_neighbors = k_neighbors
        
        # 한국어 문장 종결 패턴
        self.sentence_endings = re.compile(r'([.!?])\s+')
    
    def __call__(self, paragraph: str) -> Dict:
        """
        문단을 문장 단위로 분해하고 전처리
        
        Args:
            paragraph: 입력 문단
        
        Returns:
            {
                "sentences": List[str],
                "tokens": torch.Tensor [num_sents, seq_len],
                "replacement_mask": torch.Tensor [num_sents, seq_len],
                "topo_idx": torch.Tensor [num_sents, k],
                "metadata": Dict
            }
        """
        # 1. 문장 분해
        sentences = self._segment_sentences(paragraph)
        
        if len(sentences) == 0:
            # 빈 입력 처리
            return {
                "sentences": [],
                "tokens": torch.zeros((0, 0), dtype=torch.long),
                "replacement_mask": torch.zeros((0, 0), dtype=torch.long),
                "topo_idx": torch.zeros((0, self.k_neighbors), dtype=torch.long),
                "metadata": {"num_sentences": 0, "sentence_lengths": [], "total_tokens": 0}
            }
        
        # 2. 토큰화 (간단한 문자 단위 토큰화)
        tokens, token_strings = self._tokenize_sentences(sentences)
        
        # 3. Replacement mask 생성
        replacement_mask = self._generate_replacement_mask(token_strings, sentences)
        
        # 4. Topology index 생성
        topo_idx = self._build_topology(len(sentences), k=self.k_neighbors)
        
        # 5. 메타데이터
        metadata = {
            "num_sentences": len(sentences),
            "sentence_lengths": [len(s.split()) for s in sentences],
            "total_tokens": tokens.shape[1]
        }
        
        return {
            "sentences": sentences,
            "tokens": tokens,
            "replacement_mask": replacement_mask,
            "topo_idx": topo_idx,
            "metadata": metadata
        }
    
    def _segment_sentences(self, paragraph: str) -> List[str]:
        """
        문장 분해
        
        docs 명세: 
        - 한국어 kss 또는 nltk.sent_tokenize 사용
        - 너무 짧은 문장 병합
        """
        # 간단한 정규식 기반 문장 분리
        sentences = []
        current = []
        
        for char in paragraph:
            current.append(char)
            if char in '.!?' and len(''.join(current).strip()) > 5:
                sent = ''.join(current).strip()
                if sent:
                    sentences.append(sent)
                current = []
        
        # 남은 문자열 처리
        if current:
            sent = ''.join(current).strip()
            if sent:
                sentences.append(sent)
        
        # 후처리: 너무 짧은 문장 병합
        merged = []
        buffer = ""
        for sent in sentences:
            if len(sent) < 10 and buffer:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent
        if buffer:
            merged.append(buffer)
        
        return merged if merged else sentences
    
    def _tokenize_sentences(self, sentences: List[str]) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        문장 토큰화 (간단한 문자 단위)
        
        Returns:
            tokens: [num_sents, max_seq_len] 토큰 ID 텐서
            token_strings: [num_sents][seq_len] 토큰 문자열 리스트
        """
        all_tokens = []
        all_token_strings = []
        
        for sent in sentences:
            # 문자 단위 토큰화
            chars = list(sent)
            # 간단한 ID 매핑 (ord 사용)
            token_ids = [min(ord(c), 50000) for c in chars]  # 범위 제한
            all_tokens.append(token_ids)
            all_token_strings.append(chars)
        
        # 패딩
        max_len = min(max(len(t) for t in all_tokens) if all_tokens else 0, self.max_length)
        
        padded_tokens = []
        for tokens in all_tokens:
            padded = tokens[:max_len] + [0] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        return torch.tensor(padded_tokens, dtype=torch.long), all_token_strings
    
    def _generate_replacement_mask(
        self,
        token_strings: List[List[str]],
        sentences: List[str]
    ) -> torch.Tensor:
        """
        교체 가능 토큰 마스크 생성
        
        docs 명세:
        - 고정 토큰: 고유명사, 숫자, 특수 기호
        - 교체 가능: 일반 명사, 동사, 형용사
        
        Returns:
            mask: [num_sents, seq_len] 0=고정, 1=교체 가능
        """
        masks = []
        
        for tokens in token_strings:
            mask = []
            for token in tokens:
                # 간단한 규칙 기반 판정
                if self._is_replaceable(token):
                    mask.append(1)
                else:
                    mask.append(0)
            masks.append(mask)
        
        # 패딩
        max_len = max(len(m) for m in masks) if masks else 0
        padded_masks = []
        for mask in masks:
            padded = mask[:max_len] + [0] * (max_len - len(mask))
            padded_masks.append(padded)
        
        return torch.tensor(padded_masks, dtype=torch.long)
    
    def _is_replaceable(self, token: str) -> bool:
        """
        토큰 교체 가능 여부 판정
        
        docs 명세:
        - 특수 토큰, 조사, 어미, 숫자, 기호 제외
        - 일반 명사, 동사, 형용사 허용
        """
        # 숫자, 기호 제외
        if token.isdigit() or token in ".,!?;:()[]{}\"'":
            return False
        
        # 공백 제외
        if token.isspace():
            return False
        
        # 한글 자음/모음 단독은 조사로 간주
        if len(token) == 1 and ord('ㄱ') <= ord(token) <= ord('ㅎ'):
            return False
        if len(token) == 1 and ord('ㅏ') <= ord(token) <= ord('ㅣ'):
            return False
        
        # 기본적으로 한글/영문은 교체 가능
        if token.isalnum():
            return True
        
        # 한글 판정
        if '가' <= token <= '힣':
            return True
        
        return False
    
    def _build_topology(self, num_sentences: int, k: int = 3) -> torch.Tensor:
        """
        시간 순서 기반 topology 생성
        
        docs 명세:
        - 시간 순서: 이전/다음 문장을 이웃으로
        - k개 채우기
        
        Returns:
            topo_idx: [num_sentences, k] 이웃 인덱스
        """
        topo = []
        for i in range(num_sentences):
            neighbors = []
            
            # 이전 문장
            if i > 0:
                neighbors.append(i - 1)
            
            # 다음 문장
            if i < num_sentences - 1:
                neighbors.append(i + 1)
            
            # k개 채우기 (자기 자신으로)
            while len(neighbors) < k:
                neighbors.append(i)
            
            topo.append(neighbors[:k])
        
        return torch.tensor(topo, dtype=torch.long)

