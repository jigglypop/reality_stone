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
from experiments.lbigd.core.lbo import build_laplacian_matrix, laplacian_mul  # noqa: E402
from experiments.lbigd.core.metric import ring, update as update_metric  # noqa: E402
from experiments.lbigd.core.dopamine import DopamineGate  # noqa: E402


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


def _lowfreq_signal(n: int) -> np.ndarray:
    x = np.arange(int(n), dtype=np.float64)
    return (0.8 * np.sin(2.0 * np.pi * x / float(n)) + 0.2 * np.sin(4.0 * np.pi * x / float(n))).astype(np.float64)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def _rolling_sigma(x: np.ndarray, win: int) -> np.ndarray:
    out = np.full((x.shape[0],), np.nan, dtype=np.float64)
    if x.shape[0] < int(win) + 2:
        return out
    for t in range(int(win), x.shape[0]):
        seg = x[(t - int(win)) : t]
        out[t] = float(np.std(seg, ddof=1))
    return out


def _phase_mean(*, t: np.ndarray, valid: np.ndarray, lo: int, hi: int, values: Dict[str, np.ndarray]) -> Dict[str, float]:
    m = valid & (t >= int(lo)) & (t < int(hi))
    out: Dict[str, float] = {"n": float(np.sum(m))}
    for k, v in values.items():
        out[k] = float(np.mean(v[m])) if np.any(m) else float("nan")
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    v = np.array(vals, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def _run_collapse(
    *,
    seeds: int,
    n: int,
    steps: int,
    dt: float,
    alpha: float,
    k0: int,
    win: int,
    burn: int,
    noise_low: float,
    noise_high: float,
    t1: int,
    t2: int,
    verbose: bool,
) -> None:
    """
    CP2 tool: reproduce docs/09_intelligence/06_붕괴와_회복.md idea.
    - stable: low noise
    - collapse: high noise injection
    - recovery: low noise again
    """
    eps = 1e-8
    w = ring(int(n), weight=1.0)
    L = build_laplacian_matrix(w.astype(np.float32)).astype(np.float64)
    _, evecs = np.linalg.eigh(L)
    y = _lowfreq_signal(int(n))

    def noise_t(t: int) -> float:
        if t < int(t1):
            return float(noise_low)
        if t < int(t2):
            return float(noise_high)
        return float(noise_low)

    rows = []
    for seed in range(int(seeds)):
        rng = np.random.default_rng(int(seed))
        u = np.zeros((int(n),), dtype=np.float64)
        err = np.zeros((int(steps),), dtype=np.float64)
        r_low = np.zeros((int(steps),), dtype=np.float64)
        a1 = np.zeros((int(steps),), dtype=np.float64)

        for t in range(int(steps)):
            x = y + noise_t(t) * rng.standard_normal((int(n),), dtype=np.float64)
            du = float(alpha) * (x - u) - laplacian_mul(w, u.astype(np.float32)).astype(np.float64)
            u = u + float(dt) * du

            err[t] = _mse(u, y)
            a = u @ evecs
            e = a * a
            e_tot = float(np.sum(e)) + eps
            e_low = float(np.sum(e[1 : (min(int(k0), int(n) - 1) + 1)]))
            r_low[t] = e_low / e_tot
            a1[t] = float(a[1])

        da1 = np.diff(a1) / float(dt)  # (steps-1,)
        sig = _rolling_sigma(da1, int(win))  # (steps-1,)
        t_idx = np.arange(1, int(steps))
        err1 = err[1:]
        r1 = r_low[1:]
        I = r1 * (1.0 / (sig + eps))
        valid = np.isfinite(I)

        stable = _phase_mean(t=t_idx, valid=valid, lo=int(burn), hi=int(t1), values={"err": err1, "I": I})
        collapse = _phase_mean(t=t_idx, valid=valid, lo=int(t1), hi=int(t2), values={"err": err1, "I": I})
        recovery = _phase_mean(t=t_idx, valid=valid, lo=int(t2), hi=int(steps), values={"err": err1, "I": I})
        corr = float(np.corrcoef(I[valid], err1[valid])[0, 1]) if int(np.sum(valid)) >= 3 else float("nan")

        rows.append({"stable": stable, "collapse": collapse, "recovery": recovery, "corr": corr})
        if verbose:
            print(f"[seed={seed}] stable err={stable['err']:.6f} I={stable['I']:.6f} | collapse err={collapse['err']:.6f} I={collapse['I']:.6f} | recovery err={recovery['err']:.6f} I={recovery['I']:.6f} | corr={corr:.4f}")

    e_st_m, e_st_s = _mean_std([float(r["stable"]["err"]) for r in rows])
    e_c_m, e_c_s = _mean_std([float(r["collapse"]["err"]) for r in rows])
    e_r_m, e_r_s = _mean_std([float(r["recovery"]["err"]) for r in rows])
    I_st_m, I_st_s = _mean_std([float(r["stable"]["I"]) for r in rows])
    I_c_m, I_c_s = _mean_std([float(r["collapse"]["I"]) for r in rows])
    I_r_m, I_r_s = _mean_std([float(r["recovery"]["I"]) for r in rows])
    corr_m, corr_s = _mean_std([float(r["corr"]) for r in rows])

    print("collapse_report")
    print(f"- seeds: {int(seeds)}")
    print(f"- n={int(n)} steps={int(steps)} dt={float(dt)} alpha={float(alpha)} k0={int(k0)} win={int(win)} burn={int(burn)}")
    print(f"- noise: stable={float(noise_low)} collapse={float(noise_high)} recovery={float(noise_low)}")
    print(f"- phases: stable=[{int(burn)},{int(t1)}) collapse=[{int(t1)},{int(t2)}) recovery=[{int(t2)},{int(steps)})")
    print(f"- stable   err: {e_st_m:.6f} ± {e_st_s:.6f} | I: {I_st_m:.6f} ± {I_st_s:.6f}")
    print(f"- collapse err: {e_c_m:.6f} ± {e_c_s:.6f} | I: {I_c_m:.6f} ± {I_c_s:.6f}")
    print(f"- recovery err: {e_r_m:.6f} ± {e_r_s:.6f} | I: {I_r_m:.6f} ± {I_r_s:.6f}")
    print(f"- corr(I,err): {corr_m:.6f} ± {corr_s:.6f}")


def _run_gate(
    *,
    seeds: int,
    n: int,
    steps1: int,
    steps2: int,
    dt: float,
    alpha: float,
    noise: float,
    k0: int,
    win: int,
    burn: int,
    verbose: bool,
) -> None:
    """
    CP2 tool: reproduce "geometry shift + recovery mechanism" (Gate ON/OFF).
    """
    eps = 1e-8
    total = int(steps1) + int(steps2)

    def simulate(seed: int, gate_enabled: bool) -> Dict[str, float]:
        rng_perm = np.random.default_rng(int(seed))
        y_base = _lowfreq_signal(int(n))
        perm = rng_perm.permutation(int(n))
        y2 = y_base[perm]

        rng = np.random.default_rng(int(seed))
        w = ring(int(n), weight=1.0)
        u = np.zeros((int(n),), dtype=np.float64)

        gate = DopamineGate(ratio=1.6, hold_steps=12) if gate_enabled else None
        triggers = 0

        # metric update params
        tau = 0.35
        topk = 6
        metric_lr = 0.25
        metric_decay = 0.01
        w_max = 1.0

        err = np.zeros((total,), dtype=np.float64)
        r_low = np.zeros((total,), dtype=np.float64)
        a1 = np.zeros((total,), dtype=np.float64)

        # eigenvectors for current w (recompute only when w changes)
        L = build_laplacian_matrix(w.astype(np.float32)).astype(np.float64)
        _, evecs = np.linalg.eigh(L)
        phi1_prev = evecs[:, 1].copy()

        def refresh_modes() -> None:
            nonlocal evecs, phi1_prev
            L2 = build_laplacian_matrix(w.astype(np.float32)).astype(np.float64)
            _, ev = np.linalg.eigh(L2)
            if float(np.dot(ev[:, 1], phi1_prev)) < 0.0:
                ev[:, 1] *= -1.0
            phi1_prev = ev[:, 1].copy()
            evecs = ev

        for t in range(total):
            y = y_base if t < int(steps1) else y2
            x = y + float(noise) * rng.standard_normal((int(n),), dtype=np.float64)
            du = float(alpha) * (x - u) - laplacian_mul(w, u.astype(np.float32)).astype(np.float64)
            u = u + float(dt) * du

            err[t] = _mse(u, y)
            a = u @ evecs
            e = a * a
            e_tot = float(np.sum(e)) + eps
            e_low = float(np.sum(e[1 : (min(int(k0), int(n) - 1) + 1)]))
            r_low[t] = e_low / e_tot
            a1[t] = float(a[1])

            if not gate_enabled:
                continue

            if gate is not None and gate.update(float(err[t])):
                triggers += 1
                w = update_metric(
                    w,
                    u.astype(np.float32, copy=False),
                    lr=float(metric_lr),
                    tau=float(tau),
                    topk=int(topk),
                    decay=float(metric_decay),
                    w_max=float(w_max),
                )
                refresh_modes()

        da1 = np.diff(a1) / float(dt)
        sig = _rolling_sigma(da1, int(win))
        t_idx = np.arange(1, total)
        err1 = err[1:]
        r1 = r_low[1:]
        I = r1 * (1.0 / (sig + eps))
        valid = np.isfinite(I)

        stable = _phase_mean(t=t_idx, valid=valid, lo=int(burn), hi=int(steps1), values={"err": err1, "r": r1, "I": I})
        early = _phase_mean(t=t_idx, valid=valid, lo=int(steps1), hi=int(steps1) + 60, values={"err": err1, "r": r1, "I": I})
        tail = _phase_mean(t=t_idx, valid=valid, lo=total - 120, hi=total, values={"err": err1, "r": r1, "I": I})
        corr = float(np.corrcoef(I[valid], err1[valid])[0, 1]) if int(np.sum(valid)) >= 3 else float("nan")

        return {
            "triggers": float(triggers),
            "e_st": float(stable["err"]),
            "e_early": float(early["err"]),
            "e_tail": float(tail["err"]),
            "r_early": float(early["r"]),
            "r_tail": float(tail["r"]),
            "I_early": float(early["I"]),
            "I_tail": float(tail["I"]),
            "corr": float(corr),
        }

    rows_on = [simulate(seed=i, gate_enabled=True) for i in range(int(seeds))]
    rows_off = [simulate(seed=i, gate_enabled=False) for i in range(int(seeds))]

    imp = np.array([rows_off[i]["e_tail"] - rows_on[i]["e_tail"] for i in range(int(seeds))], dtype=np.float64)

    print("gate_report")
    print(f"- seeds: {int(seeds)}")
    print(f"- n={int(n)} steps1={int(steps1)} steps2={int(steps2)} dt={float(dt)} alpha={float(alpha)} noise={float(noise)} k0={int(k0)} win={int(win)} burn={int(burn)}")
    on_tr_m, on_tr_s = _mean_std([float(r["triggers"]) for r in rows_on])
    print(f"- gate_on triggers: {on_tr_m:.3f} ± {on_tr_s:.3f}")
    for label, rows in [("gate_on", rows_on), ("gate_off", rows_off)]:
        e_st_m, e_st_s = _mean_std([float(r["e_st"]) for r in rows])
        e_early_m, e_early_s = _mean_std([float(r["e_early"]) for r in rows])
        e_tail_m, e_tail_s = _mean_std([float(r["e_tail"]) for r in rows])
        r_early_m, r_early_s = _mean_std([float(r["r_early"]) for r in rows])
        r_tail_m, r_tail_s = _mean_std([float(r["r_tail"]) for r in rows])
        I_early_m, I_early_s = _mean_std([float(r["I_early"]) for r in rows])
        I_tail_m, I_tail_s = _mean_std([float(r["I_tail"]) for r in rows])
        corr_m, corr_s = _mean_std([float(r["corr"]) for r in rows])

        print(f"- {label} phase1 err: {e_st_m:.6f} ± {e_st_s:.6f}")
        print(f"- {label} phase2 early err: {e_early_m:.6f} ± {e_early_s:.6f} | R_low: {r_early_m:.6f} ± {r_early_s:.6f} | I: {I_early_m:.6f} ± {I_early_s:.6f}")
        print(f"- {label} phase2 tail  err: {e_tail_m:.6f} ± {e_tail_s:.6f} | R_low: {r_tail_m:.6f} ± {r_tail_s:.6f} | I: {I_tail_m:.6f} ± {I_tail_s:.6f}")
        print(f"- {label} corr(I,err): {corr_m:.6f} ± {corr_s:.6f}")

    print(f"- tail_err_improvement(off-on): {float(imp.mean()):.6f} ± {float(imp.std(ddof=1)) if imp.size>1 else 0.0:.6f}")


def _format_context(hits: List[Tuple[float, float, Chunk]], *, max_context_chars: int, chunk_chars: int) -> Tuple[str, List[str]]:
    """
    Returns:
      context_text: concatenated context blocks for prompting
      sources: list of "path :: title" strings
    """
    blocks: List[str] = []
    sources: List[str] = []
    used = 0
    for rank, (_score, _raw, c) in enumerate(hits, start=1):
        src = f"{c.path} :: {c.title}"
        sources.append(src)
        txt = c.text.strip().replace("\r\n", "\n").replace("\r", "\n")
        if int(chunk_chars) > 0 and len(txt) > int(chunk_chars):
            txt = txt[: int(chunk_chars)].rstrip() + " ..."
        block = f"[CONTEXT {rank}] {src}\n{txt}"
        if used + len(block) + 2 > int(max_context_chars):
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks).strip(), sources


def _strip_md(s: str) -> str:
    t = s.strip()
    if t.startswith("- "):
        t = t[2:].lstrip()
    if t.startswith(">"):
        t = t.lstrip(">").lstrip()
    # remove emphasis markers
    t = t.replace("**", "").replace("__", "")
    t = t.replace("`", "")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _draft_to_paragraph(query: str, draft: List[str]) -> str:
    # Simple extractive paragraph: select a few informative lines and stitch.
    if not draft:
        return "모르겠다."

    domain_keywords = [
        "저주파",
        "고주파",
        "모드",
        "스펙트럼",
        "라플라시안",
        "lbo",
        "리만",
        "다양체",
        "고유값",
        "고유함수",
        "안정성",
        "비율",
        "σ",
        "r_low",
        "i=",
        "i(",
        "지수",
        "붕괴",
        "회복",
    ]

    scored: List[Tuple[float, str]] = []
    for line in draft:
        t = _strip_md(line)
        if not t or len(t) < 8:
            continue
        low = t.lower()
        score = 0.0
        for kw in domain_keywords:
            if kw in t or kw in low:
                score += 2.0
        # prefer longer, contentful lines
        score += min(2.0, len(t) / 120.0)
        scored.append((score, t))

    if not scored:
        return " ".join([_strip_md(x) for x in draft if _strip_md(x)])[:400].rstrip() + "."

    scored.sort(key=lambda x: -x[0])
    picked: List[str] = []
    seen = set()
    for _s, t in scored:
        if t in seen:
            continue
        seen.add(t)
        picked.append(t)
        if len(picked) >= 3:
            break

    out = " ".join(picked).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _run_chat(
    *,
    roots: List[str],
    query: str,
    topk: int,
    rho: float,
    nu: float,
    adj_weight: float,
    link_weight: float,
    max_chars: int,
    model_name: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_context_chars: int,
    chunk_chars: int,
    no_llm: bool,
) -> None:
    chunks = _build_chunks(list(roots), max_chars=int(max_chars))
    if not chunks:
        print("chat_report")
        print("- error: no markdown chunks found")
        return

    w = _build_graph(chunks, adj_weight=float(adj_weight), link_weight=float(link_weight))
    hits = _search(chunks=chunks, w=w, query=str(query), topk=int(topk), rho=float(rho), nu=float(nu))
    ctx, sources = _format_context(hits, max_context_chars=int(max_context_chars), chunk_chars=int(chunk_chars))
    draft = _answer_from_hits(str(query), hits, max_items=6)

    # Always print something quickly (even if HF download/model load takes time).
    print("chat_report")
    print(f"- query: {query}")
    print(f"- sources: {len(sources)}")
    for s in sources[: max(1, int(topk))]:
        print(f"  - {s}")
    print()
    print("answer_draft:")
    if draft:
        for s in draft:
            print(f"- {s}")
    else:
        print("- 모르겠다")
    print()
    print("answer_extractive:")
    print(_draft_to_paragraph(str(query), draft))
    print()

    if bool(no_llm):
        return

    # Build an instruction-ish prompt (works best with instruct models; still usable for tiny GPT2).
    prompt_parts = [
        "너는 문서 기반 어시스턴트다.",
        "아래 CONTEXT에 있는 내용만으로 질문에 답해라.",
        "CONTEXT에 근거가 없으면 \"모르겠다\"라고 답해라.",
        "",
        ctx,
    ]
    if draft:
        prompt_parts += [
            "",
            "[DRAFT]",
            "\n".join([f"- {s}" for s in draft]),
        ]
    prompt_parts += [
        "",
        f"[QUESTION]\n{query}",
        "",
        "[ANSWER]",
    ]
    prompt = "\n".join([p for p in prompt_parts if p is not None]).strip() + "\n"

    answer_text = None
    llm_status = None
    try:
        import warnings
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        if device == "auto":
            use_cuda = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
            device = "cuda" if use_cuda else "cpu"

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        mdl.eval()
        mdl.to(device)

        torch.manual_seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        max_in = int(getattr(tok, "model_max_length", 1024) or 1024)
        if max_in > 8192:
            max_in = 1024
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=int(max_in))
        enc = {k: v.to(device) for k, v in enc.items()}

        do_sample = bool(float(temperature) > 0.0)
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None,
            "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
            # Basic anti-looping
            "repetition_penalty": 1.08,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(max(1e-6, float(temperature)))
            gen_kwargs["top_p"] = float(top_p)

        gen = mdl.generate(**enc, **gen_kwargs)

        out = tok.decode(gen[0], skip_special_tokens=True)
        # Extract only the completion after the last [ANSWER]
        if "[ANSWER]" in out:
            out = out.split("[ANSWER]", maxsplit=1)[-1]
        answer_text = out.strip()
        if answer_text:
            # Heuristic quality gate: tiny non-instruct LMs can loop or ignore Korean.
            def _has_korean(s: str) -> bool:
                return any("가" <= ch <= "힣" for ch in s)

            def _normalize_token(t: str) -> str:
                s = t.strip().lower()
                if not s:
                    return s
                s = re.sub(r"^[^0-9a-z가-힣]+|[^0-9a-z가-힣]+$", "", s)
                if len(s) <= 1:
                    return s
                # Light Korean particle stripping (heuristic, used only for gating).
                suffixes = ["으로", "에서", "까지", "부터", "에게", "한테", "라도", "이나", "이나", "만", "도", "의", "을", "를", "은", "는", "이", "가", "과", "와", "에", "로"]
                for suf in suffixes:
                    if len(s) > len(suf) + 1 and s.endswith(suf):
                        s = s[: -len(suf)]
                        break
                return s

            def _token_set(text: str) -> set[str]:
                out: set[str] = set()
                for tok in _TOKEN_RE.findall(text):
                    n = _normalize_token(tok)
                    if len(n) >= 2:
                        out.add(n)
                return out

            def _is_degenerate(ans: str, q: str) -> bool:
                # Compression ratio heuristic: repetitive text compresses extremely well.
                try:
                    import zlib

                    raw = ans.encode("utf-8", errors="ignore")
                    if len(raw) >= 200:
                        comp = zlib.compress(raw, level=6)
                        ratio = len(comp) / float(len(raw))
                        if ratio < 0.35:
                            return True
                except Exception:
                    pass

                toks = [t for t in re.split(r"\s+", ans.strip().lower()) if t]
                if len(toks) >= 30:
                    freq: Dict[str, int] = {}
                    for t in toks:
                        freq[t] = freq.get(t, 0) + 1
                    most = max(freq.values()) if freq else 0
                    if most / max(1, len(toks)) > 0.30:
                        return True
                if len(toks) >= 20 and len(set(toks)) <= 5:
                    return True

                # Repetitive bigram pattern check
                if len(toks) >= 20:
                    bigrams = list(zip(toks, toks[1:]))
                    if bigrams:
                        uniq_ratio = len(set(bigrams)) / float(len(bigrams))
                        if uniq_ratio < 0.80:
                            return True

                # If the question is Korean but answer has no Korean, it's likely unusable.
                if _has_korean(q) and not _has_korean(ans):
                    return True

                # Grounding check: require at least one "key" token from draft that's not just the query term.
                q_set = _token_set(q)
                d_text = "\n".join(draft) if draft else ""
                d_set = _token_set(d_text)
                a_set = _token_set(ans)
                key = {t for t in d_set if t not in q_set}
                key_strong = {t for t in key if len(t) >= 3}
                if key_strong and len(a_set & key_strong) == 0:
                    return True

                return False

            if _is_degenerate(answer_text, str(query)):
                llm_status = "degenerate_output"
                answer_text = None
        else:
            answer_text = None
    except Exception as e:
        llm_status = str(e)
        answer_text = None

    if answer_text is not None:
        print()
        print("answer:")
        print(answer_text)
        return

    # Fallback: keep the already-printed draft and add LLM status.
    if llm_status is not None:
        print("llm_status:")
        print(llm_status)
        print()


def main() -> int:
    _configure_stdio_utf8()
    p = argparse.ArgumentParser(description="Reality Stone assistant (search + no-download tools).")
    sub = p.add_subparsers(dest="cmd")

    # Backward-compat: old usage `python experiments/assistant.py --query ...`
    argv = sys.argv[1:]
    if not argv or (argv and argv[0].startswith("-")):
        argv = ["search"] + argv

    p_search = sub.add_parser("search", help="LBO-based doc retriever (CP0/CP1).")
    p_search.add_argument("--roots", nargs="*", default=["docs/09_intelligence", "README.md"], help="Files/dirs to index.")
    p_search.add_argument("--query", type=str, required=True, help="Search query.")
    p_search.add_argument("--topk", type=int, default=5, help="Number of results.")
    p_search.add_argument("--rho", type=float, default=1.0, help="Resolvent rho (>0).")
    p_search.add_argument("--nu", type=float, default=0.5, help="Resolvent nu (>=0).")
    p_search.add_argument("--adj-weight", type=float, default=1.0, help="Within-file adjacency edge weight.")
    p_search.add_argument("--link-weight", type=float, default=0.7, help="Markdown link edge weight.")
    p_search.add_argument("--max-chars", type=int, default=6000, help="Max characters per chunk (splits long sections).")
    p_search.add_argument("--show", type=int, default=400, help="Show up to N chars per result (0 = no text).")
    p_search.add_argument("--answer", action="store_true", help="Print an extractive answer draft from retrieved chunks.")
    p_search.add_argument("--answer-items", type=int, default=6, help="Max number of answer bullet points.")

    p_col = sub.add_parser("collapse", help="No-download collapse/recovery validation (noise injection).")
    p_col.add_argument("--seeds", type=int, default=10)
    p_col.add_argument("--n", type=int, default=64)
    p_col.add_argument("--steps", type=int, default=900)
    p_col.add_argument("--dt", type=float, default=0.15)
    p_col.add_argument("--alpha", type=float, default=2.0)
    p_col.add_argument("--k0", type=int, default=10)
    p_col.add_argument("--win", type=int, default=60)
    p_col.add_argument("--burn", type=int, default=-1, help="Burn-in start step for stable phase (default: 2*win).")
    p_col.add_argument("--noise-low", type=float, default=0.25)
    p_col.add_argument("--noise-high", type=float, default=2.0)
    p_col.add_argument("--t1", type=int, default=300, help="Collapse start step.")
    p_col.add_argument("--t2", type=int, default=600, help="Recovery start step.")
    p_col.add_argument("--verbose", action="store_true")

    p_gate = sub.add_parser("gate", help="No-download gate on/off validation (geometry shift).")
    p_gate.add_argument("--seeds", type=int, default=5)
    p_gate.add_argument("--n", type=int, default=64)
    p_gate.add_argument("--steps1", type=int, default=250)
    p_gate.add_argument("--steps2", type=int, default=350)
    p_gate.add_argument("--dt", type=float, default=0.15)
    p_gate.add_argument("--alpha", type=float, default=2.0)
    p_gate.add_argument("--noise", type=float, default=0.6)
    p_gate.add_argument("--k0", type=int, default=10)
    p_gate.add_argument("--win", type=int, default=60)
    p_gate.add_argument("--burn", type=int, default=-1, help="Burn-in start step for phase1 stats (default: 2*win).")
    p_gate.add_argument("--verbose", action="store_true")

    p_chat = sub.add_parser("chat", help="CP3: search + small HF causal LM answer (minimal LLM).")
    p_chat.add_argument("--roots", nargs="*", default=["docs/09_intelligence", "README.md"], help="Files/dirs to index.")
    p_chat.add_argument("--query", type=str, required=True, help="Question to answer.")
    p_chat.add_argument("--topk", type=int, default=5)
    p_chat.add_argument("--rho", type=float, default=1.0)
    p_chat.add_argument("--nu", type=float, default=0.5)
    p_chat.add_argument("--adj-weight", type=float, default=1.0)
    p_chat.add_argument("--link-weight", type=float, default=0.7)
    p_chat.add_argument("--max-chars", type=int, default=6000)
    p_chat.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", help="HF model name or local path.")
    p_chat.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p_chat.add_argument("--max-new-tokens", type=int, default=192)
    p_chat.add_argument("--temperature", type=float, default=0.2)
    p_chat.add_argument("--top-p", type=float, default=0.95)
    p_chat.add_argument("--seed", type=int, default=0)
    p_chat.add_argument("--max-context-chars", type=int, default=8000)
    p_chat.add_argument("--chunk-chars", type=int, default=2000)
    p_chat.add_argument("--no-llm", action="store_true", help="Skip HF model generation and only print answer_draft + sources.")

    args = p.parse_args(argv)
    cmd = str(getattr(args, "cmd", "") or "")
    if not cmd:
        p.print_help()
        return 2

    if cmd == "search":
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

    if cmd == "collapse":
        burn = int(args.burn)
        if burn < 0:
            burn = int(max(1, 2 * int(args.win)))
        _run_collapse(
            seeds=int(args.seeds),
            n=int(args.n),
            steps=int(args.steps),
            dt=float(args.dt),
            alpha=float(args.alpha),
            k0=int(args.k0),
            win=int(args.win),
            burn=int(burn),
            noise_low=float(args.noise_low),
            noise_high=float(args.noise_high),
            t1=int(args.t1),
            t2=int(args.t2),
            verbose=bool(args.verbose),
        )
        return 0

    if cmd == "gate":
        burn = int(args.burn)
        if burn < 0:
            burn = int(max(1, 2 * int(args.win)))
        _run_gate(
            seeds=int(args.seeds),
            n=int(args.n),
            steps1=int(args.steps1),
            steps2=int(args.steps2),
            dt=float(args.dt),
            alpha=float(args.alpha),
            noise=float(args.noise),
            k0=int(args.k0),
            win=int(args.win),
            burn=int(burn),
            verbose=bool(args.verbose),
        )
        return 0

    if cmd == "chat":
        _run_chat(
            roots=list(args.roots),
            query=str(args.query),
            topk=int(args.topk),
            rho=float(args.rho),
            nu=float(args.nu),
            adj_weight=float(args.adj_weight),
            link_weight=float(args.link_weight),
            max_chars=int(args.max_chars),
            model_name=str(args.model),
            device=str(args.device),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            max_context_chars=int(args.max_context_chars),
            chunk_chars=int(args.chunk_chars),
            no_llm=bool(args.no_llm),
        )
        return 0

    print("Unknown command. Use: search | collapse | gate | chat")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

