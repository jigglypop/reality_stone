from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    build_sentence_index_from_corpus,
)


class DocumentIndexer:
    
    def __init__(
        self,
        model,
        max_paragraphs: Optional[int] = None,
        auto_build: bool = False,
        corpus: Optional[str] = None,
        **kwargs
    ):
        self.model = model.model
        self.defaults = {
            "max_paragraphs": max_paragraphs or 1000,
        }
        self.defaults.update(kwargs)
        self.index: Optional[List[Dict[str, Any]]] = None
        
        if auto_build and corpus:
            self.build_index(corpus)
    
    def build_index(
        self,
        corpus_path: str,
        max_paragraphs: Optional[int] = None,
        force_rebuild: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.index is not None and not force_rebuild:
            return self.index
        
        params = {**self.defaults, **kwargs}
        max_paragraphs = max_paragraphs or params["max_paragraphs"]
        
        self.index = build_sentence_index_from_corpus(
            model=self.model,
            data_path=corpus_path,
            max_paragraphs=max_paragraphs,
        )
        
        return self.index
    
    def search(
        self,
        query: Union[str, List[str]],
        top_k: Optional[int] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        if isinstance(query, list):
            return [self.search(q, top_k=top_k, **kwargs) for q in query]
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or 5
        
        from reality_stone.utils.pre_segmenter import PreSegmenter
        from reality_stone.layers.poincare import project_to_ball, poincare_distance
        from reality_stone.layers.lorentz import from_poincare, lorentz_distance
        
        segmenter = PreSegmenter(max_length=128, k_neighbors=3)
        seg_q = segmenter(query)
        
        if seg_q["metadata"]["num_sentences"] == 0:
            return []
        
        device = next(self.model.parameters()).device
        q_tokens = seg_q["tokens"].unsqueeze(0).to(device)
        q_tokens_first = q_tokens[:, :1, :]
        
        with torch.no_grad():
            q_emb = self.model.encode_sentences(q_tokens_first)
            q_z = self.model.topic_head.poincare_embed(q_emb)
            q_z = project_to_ball(q_z)[0, 0]
        
        z_corpus = torch.stack([e["z"] for e in self.index], dim=0).to(device)
        
        c_p = float(self.model.config.c_poincare)
        N = z_corpus.shape[0]
        q_rep = q_z.unsqueeze(0).expand(N, -1)
        d_p = poincare_distance(q_rep, z_corpus, c_p)
        
        c_l = abs(float(self.model.config.c_lorentz)) if hasattr(self.model.config, "c_lorentz") else c_p
        q_l = from_poincare(q_rep, c=c_p)
        z_l = from_poincare(z_corpus, c=c_p)
        d_l = lorentz_distance(q_l, z_l, c_l)
        
        lambda_p = 0.5
        lambda_l = 0.5
        dists = lambda_p * (d_p ** 2) + lambda_l * (d_l ** 2)
        
        k = min(top_k, z_corpus.shape[0])
        topk_vals, topk_idx = torch.topk(dists, k=k, largest=False)
        
        results = []
        for rank, idx_i in enumerate(topk_idx.tolist(), start=1):
            e = self.index[idx_i]
            results.append({
                "rank": rank,
                "sentence": e["sentence"],
                "paragraph": e["paragraph"],
                "distance": float(topk_vals[rank - 1].item()),
                "metric_key": e.get("metric_key"),
            })
        
        return results
    
    def __call__(self, corpus_path: str, **kwargs):
        return self.build_index(corpus_path, **kwargs)

