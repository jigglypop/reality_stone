import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
try:
    from transformers import AutoTokenizer
except Exception: 
    AutoTokenizer = None 


@dataclass
class TreeNode:
    id: int
    type: str
    parent: Optional[int]
    text: str


@dataclass
class DocumentTree:
    nodes: List[TreeNode]
    root_id: int

    def children(self, node_id: int) -> List[int]:
        return [n.id for n in self.nodes if n.parent == node_id]


class LevelSegmenter:
    def __init__(self, level: str, parent: "PreSegmenter"):
        self.level = level
        self._segment_sentences = parent._segment_sentences
        self._tokenize_sentences = parent._tokenize_sentences
    
    def segment(self, text: str) -> List[str]:
        if self.level == "document":
            return [text]
        if self.level == "section":
            return [text]
        if self.level == "subsection":
            blocks = re.split(r"\n\n+", text)
            # Heuristic: drop leading section title block if it has no newline or sentence punctuation
            if blocks and ("\n" not in blocks[0] and "." not in blocks[0] and "!" not in blocks[0] and "?" not in blocks[0]):
                blocks = blocks[1:]
            return blocks
        if self.level == "paragraph":
            return re.split(r"\n\n+", text)
        if self.level == "sentence":
            return self._segment_sentences(text)
        if self.level == "phrase":
            return re.split(r"[.,;]+", text)
        if self.level == "token":
            return self._tokenize_sentences([text])[1][0]
        return [text]


class PreSegmenter:
    def __init__(
        self,
        max_length: int = 128,
        k_neighbors: int = 3,
        tokenizer_name: str = "klue/bert-base",
    ):
        self.max_length = max_length
        self.k_neighbors = k_neighbors

        self.sentence_endings = re.compile(r"([.!?])\s+")

        self.tokenizer = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                print(f"Warning: failed to load tokenizer '{tokenizer_name}': {e}")
                self.tokenizer = None
    
    def recursive_segment(self, text: str, levels: List[str] = ['document', 'paragraph', 'sentence', 'token']) -> DocumentTree:
        nodes = []
        node_id = 0
        parent_map = {}  # id -> children list
        
        def add_node(level: str, text: str, parent: Optional[int]) -> int:
            nonlocal node_id
            nid = node_id
            nodes.append(TreeNode(id=nid, type=level, parent=parent, text=text))
            if parent is not None:
                if parent not in parent_map:
                    parent_map[parent] = []
                parent_map[parent].append(nid)
            node_id += 1
            return nid
        
        # Start with root
        root_id = add_node('document', text, None)
        
        # Recursive build
        def build_level(current_id: int, current_text: str, level_idx: int):
            if level_idx >= len(levels) - 1:
                return
            next_level = levels[level_idx + 1]
            segmenter = LevelSegmenter(next_level, self)
            children_texts = segmenter.segment(current_text)
            for child_text in children_texts:
                if not child_text.strip():
                    continue
                child_id = add_node(next_level, child_text, current_id)
                build_level(child_id, child_text, level_idx + 1)
        
        build_level(root_id, text, 0)
        
        # Handle token level separately for all leaf nodes (sentences/phrases)
        for node in nodes:
            if node.type in ['sentence', 'phrase']:  # Assuming leaves before tokens
                tokens = self._tokenize_sentences([node.text])[1][0]
                for tok in tokens:
                    add_node('token', tok, node.id)
        
        tree = DocumentTree(nodes=nodes, root_id=root_id)
        return tree

    def __call__(self, paragraph: str) -> Dict:
        # Update to use recursive_segment
        tree = self.recursive_segment(paragraph)
        
        # Extract existing fields from tree
        sentences = [n.text for n in tree.nodes if n.type == 'sentence']
        tokens, token_strings = self._tokenize_sentences(sentences)
        replacement_mask = self._generate_replacement_mask(token_strings, sentences)
        topo_idx = self._build_topology(len(sentences), k=self.k_neighbors)
        # tree = self._build_document_tree(paragraph, sentences, token_strings) # This line is no longer needed
        
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
            "tree": tree,
            "metadata": metadata,
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
        문장 토큰화

        우선적으로 HF 토크나이저(예: klue/bert-base)를 사용하고,
        transformers가 없거나 로딩에 실패한 환경에서는 기존 문자 단위 토큰화로 fallback 한다.

        Returns:
            tokens: [num_sents, max_seq_len] 토큰 ID 텐서
            token_strings: [num_sents][seq_len] 토큰 문자열 리스트
        """
        all_tokens: List[List[int]] = []
        all_token_strings: List[List[str]] = []

        if self.tokenizer is not None:
            vocab_pad_id = self.tokenizer.pad_token_id or 0
            vocab_unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if vocab_unk_id is not None and vocab_unk_id != vocab_pad_id:
                fallback_id = vocab_unk_id
            else:
                fallback_id = 1 if vocab_pad_id == 0 else 0
            for sent in sentences:
                encoded = self.tokenizer.encode(
                    sent,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                )
                if len(encoded) == 0:
                    token_ids = [fallback_id]
                else:
                    token_ids = encoded
                token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
                all_tokens.append(token_ids)
                all_token_strings.append(token_strs)
        else:
            # 문자 단위 토큰화 (이전 구현과 동일한 fallback 경로)
            for sent in sentences:
                chars = list(sent)
                token_ids = [ord(c) for c in chars]
                all_tokens.append(token_ids)
                all_token_strings.append(chars)
            vocab_pad_id = 0

        max_len = max(1, min(max((len(t) for t in all_tokens), default=0), self.max_length))

        padded_tokens: List[List[int]] = []
        for token_ids in all_tokens:
            padded = token_ids[:max_len] + [vocab_pad_id] * (max_len - len(token_ids))
            padded_tokens.append(padded)

        if len(padded_tokens) == 0:
            return torch.zeros((0, 0), dtype=torch.long), []

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
        
        # 패딩 길이는 토큰화 길이와 동일하게 self.max_length로 클램프
        max_len = min(max(len(m) for m in masks) if masks else 0, self.max_length)
        padded_masks = []
        for mask in masks:
            padded = mask[:max_len] + [0] * (max_len - len(mask))
            padded_masks.append(padded)
        
        return torch.tensor(padded_masks, dtype=torch.long)
    
    def _is_replaceable(self, token: str) -> bool:
        """
        토큰 교체 가능 여부 판정
        
        개선: 논문 Section 6.2의 lexical space 확장
        - 한글 명사/동사 (2자 이상) 허용
        - 영어 단어 (3자 이상) 허용
        - 특수 토큰, 문장부호, 단독 조사만 제외
        """
        if not token or not token.strip():
            return False
        
        # 문장부호 제외
        if token in ".,!?;:()[]{}\"'":
            return False
        
        # BERT 특수 토큰 제외
        if token.startswith('[') and token.endswith(']'):
            return False
        if token.startswith('##'):
            return True
        
        # 한글 단어 (2자 이상) 허용
        if len(token) >= 2:
            has_hangul = any('가' <= c <= '힣' for c in token)
            if has_hangul:
                return True
        
        # 영어 단어 (3자 이상) 허용
        if token.isalpha() and len(token) >= 3:
            return True
        
        # 영문+숫자 혼합 (4자 이상) 허용
        if token.isalnum() and len(token) >= 4:
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

    def _build_document_tree(
        self,
        paragraph: str,
        sentences: List[str],
        token_strings: List[List[str]],
    ) -> DocumentTree:
        """
        문단을 일반 트리(document → sentence → token)로 표현.
        
        - docs 2장, 10장에서 정의한 트리 T=(V,E) 의 최소 구현:
          * type(v) ∈ {document, sentence, token}
        - 현재는 3레벨 구조이지만, 타입/노드 정의만 추가하면
          section/subsection/phrase 등으로 확장 가능.
        """
        nodes: List[TreeNode] = []
        node_id = 0

        # 루트 노드 (document)
        root_id = node_id
        nodes.append(
            TreeNode(
                id=root_id,
                type="document",
                parent=None,
                text=paragraph,
            )
        )
        node_id += 1

        # 문장 노드들
        sentence_node_ids: List[int] = []
        for sent in sentences:
            sid = node_id
            nodes.append(
                TreeNode(
                    id=sid,
                    type="sentence",
                    parent=root_id,
                    text=sent,
                )
            )
            sentence_node_ids.append(sid)
            node_id += 1

        # 토큰 노드들 (패딩 제외)
        for s_idx, sent_tokens in enumerate(token_strings):
            parent_sid = sentence_node_ids[s_idx]
            for tok in sent_tokens:
                # HF 토크나이저가 없는 경우 문자 단위 토큰도 그대로 사용
                if tok is None:
                    continue
                # 패딩 토큰(예: [PAD])은 텍스트 의미가 없으므로 노드로 만들지 않음
                if isinstance(tok, str) and tok.strip() == "":
                    continue
                tid = node_id
                nodes.append(
                    TreeNode(
                        id=tid,
                        type="token",
                        parent=parent_sid,
                        text=str(tok),
                    )
                )
                node_id += 1

        return DocumentTree(nodes=nodes, root_id=root_id)

