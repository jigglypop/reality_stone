from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class DocumentIndexer:
    model: object | None = None
    chunk_size: int = 512
    documents: List[str] = field(default_factory=list)

    def add(self, document: str) -> int:
        self.documents.append(str(document))
        return len(self.documents) - 1

    def extend(self, documents: Iterable[str]) -> list[int]:
        return [self.add(doc) for doc in documents]

    def __call__(self, documents):
        if isinstance(documents, str):
            return self.add(documents)
        return self.extend(documents)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        terms = set(str(query).lower().split())
        scored = []
        for idx, doc in enumerate(self.documents):
            words = set(doc.lower().split())
            score = len(terms & words) / max(len(terms), 1)
            scored.append({"id": idx, "text": doc, "score": float(score)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]
