"""API schemas - Phase 5

docs/sentence_topic_architecture.md의 7.2 API 설계 명세 준수
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class RewriteRequest(BaseModel):
    """
    docs 명세 7.2.2 Request Schema
    """
    paragraph: str
    lexical_overrides: Optional[Dict[str, List[str]]] = {}
    metric_hint: Optional[str] = None
    options: Optional[Dict] = {}


class RewriteResponse(BaseModel):
    """
    docs 명세 7.2.3 Response Schema
    """
    sentences: List[str]
    topics: List[List[float]]
    metric_keys: List[str]
    replacements: List[Dict]
    final_text: str
    stats: Dict

