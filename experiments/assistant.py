from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Ensure repo root is on sys.path when running as a script (python experiments/assistant.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lbigd.core.lbo import solve_resolvent  # noqa: E402


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: int
    path: str
    title: str
    level: int
    text: str
    tokens: Dict[str, int]
    norm: float


def _configure_stdio_utf8() -> None:
    # Cursor/Git Bash capture expects UTF-8. Force stdout/stderr to UTF-8 to avoid mojibake.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _tokenize(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text):
        t = tok.lower()
        counts[t] = counts.get(t, 0) + 1
    return counts


def _norm(counts: Dict[str, int]) -> float:
    if not counts:
        return 0.0
    return float(np.sqrt(sum(float(v * v) for v in counts.values())))


def _cosine(a: Dict[str, int], a_norm: float, b: Dict[str, int], b_norm: float) -> float:
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
        a_norm, b_norm = b_norm, a_norm
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        dot += float(va * vb)
    return float(dot / (a_norm * b_norm))


def _split_markdown_into_sections(text: str) -> List[Tuple[str, int, str]]:
    """
    Returns: list of (title, level, section_text).
    """
    lines = text.splitlines()
    starts: List[Tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        starts.append((i, level, title))

    if not starts:
        return [("document", 1, text)]

    sections: List[Tuple[str, int, str]] = []
    for idx, (start_i, level, title) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start_i:end_i]).strip()
        if not body:
            continue
        sections.append((title or "section", level, body))
    return sections


def _is_tiny_section(section_text: str) -> bool:
    s = section_text.strip()
    if not s:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return True
    # Pure heading-only chunk (common for top-level title immediately followed by a subheading).
    if len(lines) <= 2 and all(ln.startswith("#") for ln in lines):
        return True
    if len(s) < 80 and all(ln.startswith("#") for ln in lines):
        return True
    return False


def _merge_tiny_sections(sections: List[Tuple[str, int, str]]) -> List[Tuple[str, int, str]]:
    if len(sections) <= 1:
        return sections
    out: List[Tuple[str, int, str]] = []
    i = 0
    while i < len(sections):
        title, level, body = sections[i]
        if _is_tiny_section(body) and i + 1 < len(sections):
            n_title, n_level, n_body = sections[i + 1]
            merged_title = f"{title} / {n_title}".strip()
            merged_level = int(min(level, n_level))
            merged_body = (body.rstrip() + "\n\n" + n_body.lstrip()).strip()
            out.append((merged_title, merged_level, merged_body))
            i += 2
            continue
        out.append((title, level, body))
        i += 1
    return out


def _split_long_section(title: str, level: int, section_text: str, *, max_chars: int) -> List[Tuple[str, int, str]]:
    if len(section_text) <= max_chars:
        return [(title, level, section_text)]
    parts = re.split(r"\n\n+", section_text)
    out: List[Tuple[str, int, str]] = []
    buf: List[str] = []
    buf_len = 0
    part_idx = 1
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if buf_len + len(p) + 2 > max_chars and buf:
            out.append((f"{title} [{part_idx}]", level, "\n\n".join(buf)))
            part_idx += 1
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += len(p) + 2
    if buf:
        out.append((f"{title} [{part_idx}]", level, "\n\n".join(buf)))
    return out


def _extract_sentences(text: str) -> List[str]:
    # Cheap, language-agnostic splitter: paragraph -> sentence-ish fragments.
    parts: List[str] = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        for s in _SENT_SPLIT_RE.split(block):
            t = s.strip()
            if not t:
                continue
            # Avoid extremely long blocks (e.g., code fences)
            if len(t) > 400:
                t = t[:400].rstrip() + " ..."
            parts.append(t)
    return parts


def _answer_from_hits(query: str, hits: List[Tuple[float, float, Chunk]], *, max_items: int) -> List[str]:
    q = _tokenize(query)
    seen: set[str] = set()
    matched: List[Tuple[float, str]] = []
    fill: List[Tuple[float, str]] = []
    for score, _raw, c in hits:
        sents = _extract_sentences(c.text)
        kept = 0
        for sent in sents:
            s = sent.strip()
            if not s:
                continue
            # Avoid headings / code fences: answer should be sentence-level content.
            if s.startswith("#"):
                continue
            if s.startswith("```") or s.endswith("```"):
                continue
            if len(s) < 20:
                continue

            toks = _tokenize(s)
            overlap = 0.0
            if s in seen:
                continue
            if q:
                for k, v in q.items():
                    if k in toks:
                        overlap += float(min(int(v), int(toks[k])))

            if overlap > 0.0:
                seen.add(s)
                matched.append((overlap * (1.0 + float(score)), s))
                kept += 1
                continue

            # If we didn't match query tokens, still keep a few lead sentences
            # from high-scoring chunks as a fallback.
            if kept < 3:
                seen.add(s)
                fill.append(((1.0 + float(score)) / float(kept + 1), s))
                kept += 1

    matched.sort(key=lambda x: -x[0])
    fill.sort(key=lambda x: -x[0])

    out: List[str] = []
    for _v, s in matched:
        out.append(s)
        if len(out) >= int(max_items):
            return out
    for _v, s in fill:
        if s in out:
            continue
        out.append(s)
        if len(out) >= int(max_items):
            break
    return out


def _iter_markdown_files(roots: List[str]) -> Iterable[Path]:
    for raw in roots:
        p = (ROOT / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
        if p.is_file() and p.suffix.lower() == ".md":
            yield p
            continue
        if p.is_file():
            # allow README.md-like explicit files even if suffix isn't .md
            yield p
            continue
        if p.is_dir():
            for f in sorted(p.rglob("*.md")):
                yield f


def _build_chunks(roots: List[str], *, max_chars: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for path in _iter_markdown_files(roots):
        try:
            text = _read_text(path)
        except OSError:
            continue

        rel = str(path.relative_to(ROOT)).replace("\\", "/") if str(path).startswith(str(ROOT)) else str(path)
        sections = _merge_tiny_sections(_split_markdown_into_sections(text))
        for title, level, body in sections:
            for t2, l2, body2 in _split_long_section(title, level, body, max_chars=int(max_chars)):
                toks = _tokenize(body2)
                nrm = _norm(toks)
                chunks.append(Chunk(id=cid, path=rel, title=t2, level=l2, text=body2, tokens=toks, norm=nrm))
                cid += 1
    return chunks


def _resolve_md_link(base_path: str, target: str) -> str | None:
    # base_path is repo-relative if possible
    base = (ROOT / base_path).resolve()
    cand = (base.parent / target).resolve()
    try:
        rel = cand.relative_to(ROOT)
        return str(rel).replace("\\", "/")
    except Exception:
        return None


def _build_graph(chunks: List[Chunk], *, adj_weight: float, link_weight: float) -> np.ndarray:
    n = len(chunks)
    w = np.zeros((n, n), dtype=np.float32)
    if n == 0:
        return w

    # File adjacency (local continuity)
    by_path: Dict[str, List[int]] = {}
    for c in chunks:
        by_path.setdefault(c.path, []).append(c.id)
    for ids in by_path.values():
        ids_sorted = sorted(ids)
        for a, b in zip(ids_sorted, ids_sorted[1:]):
            w[a, b] = max(w[a, b], float(adj_weight))
            w[b, a] = max(w[b, a], float(adj_weight))

    # Link edges (explicit references)
    first_chunk_by_path: Dict[str, int] = {}
    for path, ids in by_path.items():
        first_chunk_by_path[path] = int(min(ids))

    for c in chunks:
        for m in _MD_LINK_RE.finditer(c.text):
            target = (m.group(1) or "").strip()
            if not target:
                continue
            resolved = _resolve_md_link(c.path, target)
            if not resolved:
                continue
            dst = first_chunk_by_path.get(resolved)
            if dst is None:
                continue
            if dst == c.id:
                continue
            w[c.id, dst] = max(w[c.id, dst], float(link_weight))
            w[dst, c.id] = max(w[dst, c.id], float(link_weight))

    np.fill_diagonal(w, 0.0)
    return w


def _search(
    *,
    chunks: List[Chunk],
    w: np.ndarray,
    query: str,
    topk: int,
    rho: float,
    nu: float,
) -> List[Tuple[float, float, Chunk]]:
    q_tokens = _tokenize(query)
    q_norm = _norm(q_tokens)

    raw = np.zeros((len(chunks),), dtype=np.float32)
    for c in chunks:
        raw[c.id] = float(_cosine(q_tokens, q_norm, c.tokens, c.norm))

    # LBO resolvent smoothing: propagate relevance along the doc graph
    smoothed = solve_resolvent(w, raw, rho=float(rho), nu=float(nu), kappa=0.0).astype(np.float32)

    idx = np.argsort(-smoothed)[: max(1, int(topk))]
    out: List[Tuple[float, float, Chunk]] = []
    for i in idx:
        c = chunks[int(i)]
        out.append((float(smoothed[i]), float(raw[i]), c))
    return out


def main() -> int:
    _configure_stdio_utf8()
    p = argparse.ArgumentParser(description="LBO-based doc retriever (CP0).")
    p.add_argument("--roots", nargs="*", default=["docs/09_intelligence", "README.md"], help="Files/dirs to index.")
    p.add_argument("--query", type=str, required=True, help="Search query.")
    p.add_argument("--topk", type=int, default=5, help="Number of results.")
    p.add_argument("--rho", type=float, default=1.0, help="Resolvent rho (>0).")
    p.add_argument("--nu", type=float, default=0.5, help="Resolvent nu (>=0).")
    p.add_argument("--adj-weight", type=float, default=1.0, help="Within-file adjacency edge weight.")
    p.add_argument("--link-weight", type=float, default=0.7, help="Markdown link edge weight.")
    p.add_argument("--max-chars", type=int, default=6000, help="Max characters per chunk (splits long sections).")
    p.add_argument("--show", type=int, default=400, help="Show up to N chars per result (0 = no text).")
    p.add_argument("--answer", action="store_true", help="Print an extractive answer draft from retrieved chunks.")
    p.add_argument("--answer-items", type=int, default=6, help="Max number of answer bullet points.")
    args = p.parse_args()

    chunks = _build_chunks(list(args.roots), max_chars=int(args.max_chars))
    if not chunks:
        print("No markdown chunks found.")
        return 2

    w = _build_graph(chunks, adj_weight=float(args.adj_weight), link_weight=float(args.link_weight))

    hits = _search(
        chunks=chunks,
        w=w,
        query=str(args.query),
        topk=int(args.topk),
        rho=float(args.rho),
        nu=float(args.nu),
    )

    print(f"indexed_chunks={len(chunks)} query={args.query!r}")
    if bool(args.answer):
        ans = _answer_from_hits(str(args.query), hits, max_items=int(args.answer_items))
        if ans:
            print()
            print("answer_draft:")
            for s in ans:
                print(f"- {s}")
            print()
    for rank, (score, raw, c) in enumerate(hits, start=1):
        print(f"[{rank}] score={score:.4f} raw={raw:.4f} {c.path} :: {c.title}")
        if int(args.show) > 0:
            snippet = c.text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(snippet) > int(args.show):
                snippet = snippet[: int(args.show)].rstrip() + " ..."
            print(snippet)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

