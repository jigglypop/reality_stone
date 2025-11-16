from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    answer_question_from_corpus,
    answer_question_with_llm,
)


class QuestionAnswerer:
    
    def __init__(
        self,
        model,
        corpus: Optional[str] = None,
        top_k: Optional[int] = None,
        use_llm: bool = False,
        max_paragraphs: Optional[int] = None,
        **kwargs
    ):
        self.model = model.model
        self.defaults = {
            "corpus": corpus,
            "top_k": top_k or 3,
            "use_llm": use_llm,
            "max_paragraphs": max_paragraphs or 1000,
        }
        self.defaults.update(kwargs)
    
    def __call__(
        self,
        question: Union[str, List[str]],
        context: Optional[str] = None,
        **kwargs
    ):
        if isinstance(question, list):
            return self.batch(question, context=context, **kwargs)
        
        params = {**self.defaults, **kwargs}
        corpus_path = context if context else params["corpus"]
        
        if not corpus_path:
            raise ValueError("corpus or context must be provided")
        
        if params["use_llm"]:
            result = answer_question_with_llm(
                model=self.model,
                question=question,
                data_path=corpus_path,
                top_k=params["top_k"],
                max_paragraphs=params["max_paragraphs"],
                max_new_tokens=params.get("max_new_tokens", 32),
            )
            return {
                "question": result["question"],
                "answer": result["answer"],
                "support": result["support"],
            }
        else:
            result = answer_question_from_corpus(
                model=self.model,
                question=question,
                data_path=corpus_path,
                top_k=params["top_k"],
                max_paragraphs=params["max_paragraphs"],
            )
            return {
                "question": result["question"],
                "answers": result["answers"],
                "support": result["support"],
            }
    
    def batch(
        self,
        questions: List[str],
        context: Optional[str] = None,
        **kwargs
    ):
        return [self(q, context=context, **kwargs) for q in questions]

