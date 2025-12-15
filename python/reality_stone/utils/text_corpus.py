from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class CorpusDoc:
    path: str
    text: str


def iter_text_files(
    root: str | Path,
    exts: Sequence[str] = (".md", ".txt"),
    max_bytes: int = 2_000_000,
) -> Iterator[Path]:
    root_p = Path(root)
    if root_p.is_file():
        if root_p.suffix.lower() in set(e.lower() for e in exts):
            yield root_p
        return
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in set(e.lower() for e in exts):
            continue
        try:
            if p.stat().st_size <= max_bytes:
                yield p
        except OSError:
            continue


def read_text_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return None
    # naive encoding fallback chain
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def load_corpus(
    roots: Sequence[str | Path],
    exts: Sequence[str] = (".md", ".txt"),
    max_docs: int = 2000,
    max_bytes_per_doc: int = 2_000_000,
) -> List[CorpusDoc]:
    out: List[CorpusDoc] = []
    seen: set[str] = set()
    for r in roots:
        for p in iter_text_files(r, exts=exts, max_bytes=max_bytes_per_doc):
            ps = str(p)
            if ps in seen:
                continue
            seen.add(ps)
            txt = read_text_file(p)
            if not txt:
                continue
            out.append(CorpusDoc(path=ps, text=txt))
            if len(out) >= max_docs:
                return out
    return out


def chunk_text(
    text: str,
    chunk_chars: int = 8000,
    overlap_chars: int = 1000,
) -> Iterator[str]:
    if chunk_chars <= 0:
        yield text
        return
    step = max(1, int(chunk_chars) - int(overlap_chars))
    n = len(text)
    i = 0
    while i < n:
        yield text[i : i + chunk_chars]
        i += step


