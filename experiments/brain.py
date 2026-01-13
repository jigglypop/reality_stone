from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script (python experiments/brain.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nilearn import datasets  # noqa: E402
from nilearn.maskers import NiftiLabelsMasker  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402

from experiments.lbigd.core.lbo import build_laplacian_matrix  # noqa: E402


@dataclass(frozen=True)
class SubjectResult:
    subject: int
    iq: float
    age: float | None
    mean_fd: float | None
    n_time: int
    n_rois: int
    k0: int
    r_low: float
    sigma_da1: float
    I: float


def _subject_id_from_func_path(func_path: str) -> int:
    # .../adhd/data/<ID>/<ID>_rest_....nii.gz
    sid = os.path.basename(os.path.dirname(func_path))
    return int(sid)  # drops leading zeros to match phenotypic CSV


def _default_adhd_dir() -> Path:
    # nilearn default is usually ~/nilearn_data; dataset is stored under <base>/adhd
    return Path.home() / "nilearn_data" / "adhd"


def _adhd_ids_ordered() -> list[str]:
    # Keep consistent with nilearn's fetch_adhd ordering so URL mapping is correct.
    from nilearn.datasets.func import adhd_ids

    return list(adhd_ids())


def _adhd_nitrc_ids() -> list[int]:
    # From nilearn.datasets.func.fetch_adhd: nitrc_ids = range(7782, 7822)
    return list(range(7782, 7822))


def _adhd_archive_url(nitrc_id: int, subject_id: str) -> str:
    return f"https://www.nitrc.org/frs/download.php/{int(nitrc_id)}/adhd40_{subject_id}.tgz"


def _load_phenotypic(adhd_root: str) -> pd.DataFrame:
    path = os.path.join(adhd_root, "ADHD200_40subs_motion_parameters_and_phenotypics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"phenotypic CSV not found: {path}")
    return pd.read_csv(path)


def _load_subject_id_order(adhd_dir: Path) -> list[int]:
    path = adhd_dir / "ADHD200_40subs_ID.txt"
    if not path.exists():
        return []
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.isdigit():
            continue
        ids.append(int(s))
    return ids


def _pick_iq(row: pd.Series) -> float | None:
    # Prefer full_4_iq then full_2_iq
    v4 = row.get("full_4_iq")
    if pd.notna(v4):
        return float(v4)
    v2 = row.get("full_2_iq")
    if pd.notna(v2):
        return float(v2)
    return None


def _to_optional_float(x) -> float | None:
    return None if pd.isna(x) else float(x)


def _load_confounds(path: str) -> pd.DataFrame:
    # Tab-delimited file
    df = pd.read_csv(path, sep="\t")
    df = df.select_dtypes(include=["number"]).copy()
    return df


def _remove_constant_rois(ts: np.ndarray, *, var_eps: float = 1e-12) -> np.ndarray:
    if ts.ndim != 2:
        raise ValueError("ts must be 2D (T, R)")
    var = ts.var(axis=0)
    keep = var > float(var_eps)
    return ts[:, keep]


def _cleanup_partial_downloads(adhd_dir: Path) -> int:
    # nilearn stores in-progress downloads as *.part files; if a download is interrupted,
    # the leftover can later trigger CRC/abort. Safe to remove.
    if not adhd_dir.exists():
        return 0
    removed = 0
    for p in adhd_dir.rglob("*.part"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


class _ProgressReader(io.RawIOBase):
    def __init__(self, raw, bar: tqdm):
        self._raw = raw
        self._bar = bar

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if data:
            self._bar.update(len(data))
        return data


def _safe_write_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo, out_root: Path) -> None:
    # Extract a single file member into out_root, preventing path traversal.
    rel = Path(member.name)
    target = (out_root / rel).resolve()
    root = out_root.resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"unsafe tar member path: {member.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    src = tf.extractfile(member)
    if src is None:
        raise ValueError(f"cannot extract member: {member.name}")
    with src:
        with open(target, "wb") as f:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    # Basic integrity check (member.size is size of stored file entry).
    if target.stat().st_size != int(member.size):
        try:
            target.unlink()
        except OSError:
            pass
        raise IOError(f"incomplete extract for {member.name}: expected={member.size} got={target.stat().st_size}")


def _subject_extracted(adhd_dir: Path, subject_id: str) -> bool:
    base = adhd_dir / "data" / subject_id
    return (base / f"{subject_id}_rest_tshift_RPI_voreg_mni.nii.gz").exists() and (base / f"{subject_id}_regressors.csv").exists()


def _download_and_extract_subject(
    *,
    adhd_dir: Path,
    subject_id: str,
    nitrc_id: int,
    retries: int,
) -> bool:
    if _subject_extracted(adhd_dir, subject_id):
        return True

    url = _adhd_archive_url(nitrc_id, subject_id)
    wanted = {
        f"data/{subject_id}/{subject_id}_rest_tshift_RPI_voreg_mni.nii.gz",
        f"data/{subject_id}/{subject_id}_regressors.csv",
    }

    import requests

    attempts = max(1, int(retries) + 1)
    for i in range(attempts):
        # Clean partial extracted files if any
        base = adhd_dir / "data" / subject_id
        try:
            (base / f"{subject_id}_rest_tshift_RPI_voreg_mni.nii.gz").unlink()
        except OSError:
            pass
        try:
            (base / f"{subject_id}_regressors.csv").unlink()
        except OSError:
            pass

        try:
            with requests.get(url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                bar = tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                    desc=f"dl {subject_id}",
                )
                try:
                    reader = _ProgressReader(r.raw, bar)
                    with tarfile.open(fileobj=reader, mode="r|gz") as tf:
                        for member in tf:
                            if not member.isfile():
                                continue
                            if member.name in wanted:
                                _safe_write_from_tar(tf, member, adhd_dir)
                finally:
                    bar.close()

            return _subject_extracted(adhd_dir, subject_id)
        except Exception as e:
            wait_s = 2.0 * (i + 1)
            tqdm.write(f"[dl] failed subject={subject_id} attempt={i+1}/{attempts} err={type(e).__name__}: {e} (wait {wait_s:.0f}s)")
            time.sleep(wait_s)

    return False


def _collect_local_subject_files(adhd_dir: Path) -> dict[int, tuple[str, str]]:
    # Gather any extracted subject files under the dataset dir (covers both canonical
    # adhd/data/<ID>/... and hashed cache subdirs).
    mapping: dict[int, tuple[str, str]] = {}

    def _prefer(path: Path) -> int:
        s = str(path).replace("\\", "/")
        return 0 if "/adhd/data/" in s else 1

    for func_path in adhd_dir.rglob("*_rest_tshift_RPI_voreg_mni.nii.gz"):
        subj_dir = func_path.parent
        sid_str = subj_dir.name
        if not sid_str.isdigit():
            continue
        conf_path = subj_dir / f"{sid_str}_regressors.csv"
        if not conf_path.exists():
            continue

        sid = int(sid_str)
        if sid not in mapping or _prefer(func_path) < _prefer(Path(mapping[sid][0])):
            mapping[sid] = (str(func_path), str(conf_path))
    return mapping


def _infer_tr(func_path: str, *, fallback: float = 2.0) -> float:
    try:
        import nibabel as nib

        img = nib.load(func_path)
        zooms = img.header.get_zooms()
        if len(zooms) >= 4 and float(zooms[3]) > 0:
            return float(zooms[3])
    except Exception:
        pass
    return float(fallback)


def _build_weights_from_timeseries(ts: np.ndarray) -> np.ndarray:
    # Simple functional-connectivity graph: |corr|
    c = np.corrcoef(ts, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.abs(c)
    np.fill_diagonal(w, 0.0)
    return w


def compute_intelligence_index(
    *,
    func_path: str,
    confounds_path: str,
    masker: NiftiLabelsMasker,
    t_r: float,
    k0: int,
    eps: float,
) -> dict:
    conf = _load_confounds(confounds_path)
    ts = masker.fit_transform(func_path, confounds=conf)  # (T, R)

    ts = _remove_constant_rois(ts)
    if ts.shape[1] < 3:
        return {"ok": False, "reason": "too_few_rois", "n_time": int(ts.shape[0]), "n_rois": int(ts.shape[1])}

    w = _build_weights_from_timeseries(ts)

    L = build_laplacian_matrix(w.astype(np.float32))
    evals, evecs = np.linalg.eigh(L.astype(np.float64))
    _ = evals  # silence unused; kept for potential debugging

    a = ts @ evecs  # (T, R)
    e = a * a

    # Low-frequency energy ratio over time (exclude k=0 constant-like mode)
    k0_eff = int(min(int(k0), e.shape[1] - 1))
    if k0_eff < 1:
        return {"ok": False, "reason": "k0_too_small", "n_time": int(ts.shape[0]), "n_rois": int(ts.shape[1])}

    e_low = e[:, 1 : (k0_eff + 1)].sum(axis=1)
    e_tot = e.sum(axis=1)
    r_low_t = e_low / (e_tot + float(eps))
    r_low = float(np.mean(r_low_t))

    # Stability term from first non-trivial mode (k=1): sigma(d a1 / dt)
    a1 = a[:, 1]
    da1 = np.diff(a1) / float(t_r)
    if da1.size < 2:
        return {"ok": False, "reason": "too_few_timepoints", "n_time": int(ts.shape[0]), "n_rois": int(ts.shape[1])}

    sigma = float(np.std(da1, ddof=1))
    stability = 1.0 / (sigma + float(eps))
    I = r_low * stability

    return {
        "ok": True,
        "n_time": int(ts.shape[0]),
        "n_rois": int(ts.shape[1]),
        "k0": int(k0_eff),
        "r_low": float(r_low),
        "sigma_da1": float(sigma),
        "I": float(I),
    }


def _silence_nilearn_confounds_warning() -> None:
    # Keep output focused on intermediate I values (tqdm.write).
    # nilearn 0.13 emits a deprecation warning about confounds standardization.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*confounds will be standardized using the sample std.*",
        category=DeprecationWarning,
    )


def _iter_subjects(func_paths: list[str], conf_paths: list[str]) -> Iterable[tuple[str, str]]:
    if len(func_paths) != len(conf_paths):
        raise ValueError("func and confounds lists must be same length")
    return zip(func_paths, conf_paths)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate docs/09_intelligence intelligence equation on real fMRI + IQ.")
    p.add_argument("--n-subjects", type=int, default=40, help="Number of ADHD200 subjects to fetch/analyze (max 40).")
    p.add_argument("--k0", type=int, default=10, help="Low-frequency cutoff (k<=k0), excluding k=0.")
    p.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon to avoid divide-by-zero.")
    p.add_argument(
        "--atlas",
        type=str,
        default="schaefer100",
        choices=["schaefer100", "harvardoxford", "msdl"],
        help="ROI atlas used to extract time series. Avoids AAL due to SSL issues on some hosts.",
    )
    p.add_argument(
        "--fetch-retries",
        type=int,
        default=3,
        help="Retries per subject download.",
    )
    args = p.parse_args()

    _silence_nilearn_confounds_warning()

    n_subjects = int(args.n_subjects)
    if n_subjects < 1 or n_subjects > 40:
        raise SystemExit("--n-subjects must be in [1, 40]")

    # Public dataset (ADHD200 sample) via nilearn (best-effort download)
    adhd_dir = _default_adhd_dir()
    removed = _cleanup_partial_downloads(adhd_dir)
    if removed:
        print(f"[fetch] removed leftover partial downloads: {removed}")

    # Download missing subjects robustly (avoid nilearn's *.part rename issues on Windows)
    all_ids = _adhd_ids_ordered()
    all_nitrc = _adhd_nitrc_ids()
    wanted_pairs = list(zip(all_ids[:n_subjects], all_nitrc[:n_subjects], strict=False))

    print(f"[fetch] ensuring subjects downloaded: target={n_subjects}")
    for sid_str, nitrc_id in tqdm(wanted_pairs, desc="download", unit="subj"):
        ok = _download_and_extract_subject(
            adhd_dir=adhd_dir,
            subject_id=str(sid_str),
            nitrc_id=int(nitrc_id),
            retries=int(args.fetch_retries),
        )
        if not ok:
            tqdm.write(f"[dl] giving up subject={sid_str}")

    print("[data] scanning local subject files...")
    local = _collect_local_subject_files(adhd_dir)
    if not local:
        raise SystemExit(f"No local ADHD200 subject files found under: {adhd_dir}")
    print(f"[data] found extracted subjects: {len(local)}")

    # Subject selection order: align with nilearn's canonical ordering
    wanted = [int(s) for s in _adhd_ids_ordered()[:n_subjects]]

    # Infer TR from first available subject file
    first_sid = next((sid for sid in wanted if sid in local), next(iter(local.keys())))
    t_r = _infer_tr(local[first_sid][0], fallback=2.0)
    print(f"[data] inferred TR={t_r:g}s from subject={first_sid}")

    # Atlas (downloadable via nilearn; pick hosts that work reliably in this environment)
    print(f"[atlas] fetching atlas: {args.atlas}")
    if args.atlas == "schaefer100":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0)
        labels_img = atlas.maps
    elif args.atlas == "harvardoxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True, verbose=0)
        labels_img = atlas.maps
    elif args.atlas == "msdl":
        atlas = datasets.fetch_atlas_msdl(verbose=0)
        labels_img = atlas.maps
    else:
        raise SystemExit(f"unsupported atlas: {args.atlas}")

    masker = NiftiLabelsMasker(
        labels_img=labels_img,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=t_r,
        verbose=0,
    )

    # Phenotypic table
    ph = _load_phenotypic(str(adhd_dir))
    print(f"[phenotypic] loaded rows={ph.shape[0]}")

    results: list[SubjectResult] = []
    skipped: dict[str, int] = {}

    pairs: list[tuple[str, str]] = []
    missing_files = 0
    for sid in wanted:
        pair = local.get(int(sid))
        if pair is None:
            missing_files += 1
            continue
        pairs.append(pair)

    if missing_files:
        print(f"[data] missing subject files for {missing_files}/{len(wanted)} requested subjects (running on {len(pairs)}).")

    it = _iter_subjects([p[0] for p in pairs], [p[1] for p in pairs])
    for func_path, conf_path in tqdm(it, total=len(pairs), desc="subjects", unit="subj"):
        sid = _subject_id_from_func_path(func_path)

        hit = ph.loc[ph["Subject"] == sid]
        if hit.shape[0] != 1:
            skipped["phenotypic_miss"] = skipped.get("phenotypic_miss", 0) + 1
            continue

        row = hit.iloc[0]
        iq = _pick_iq(row)
        if iq is None:
            skipped["missing_iq"] = skipped.get("missing_iq", 0) + 1
            continue

        out = compute_intelligence_index(
            func_path=func_path,
            confounds_path=conf_path,
            masker=masker,
            t_r=t_r,
            k0=int(args.k0),
            eps=float(args.eps),
        )
        if not out.get("ok"):
            key = str(out.get("reason") or "compute_failed")
            skipped[key] = skipped.get(key, 0) + 1
            tqdm.write(f"[skip] subject={sid} reason={key} n_time={out.get('n_time')} n_rois={out.get('n_rois')}")
            continue

        res = SubjectResult(
            subject=sid,
            iq=float(iq),
            age=_to_optional_float(row.get("age")),
            mean_fd=_to_optional_float(row.get("MeanFD")),
            n_time=int(out["n_time"]),
            n_rois=int(out["n_rois"]),
            k0=int(out["k0"]),
            r_low=float(out["r_low"]),
            sigma_da1=float(out["sigma_da1"]),
            I=float(out["I"]),
        )
        results.append(res)

        tqdm.write(
            " ".join(
                [
                    f"[ok] subject={res.subject}",
                    f"iq={res.iq:.1f}",
                    f"I={res.I:.4f}",
                    f"r_low={res.r_low:.4f}",
                    f"sigma_da1={res.sigma_da1:.6f}",
                    f"rois={res.n_rois}",
                    f"T={res.n_time}",
                ]
            )
        )

    if not results:
        print("No usable subjects after filtering.")
        if skipped:
            print("skipped", skipped)
        return 2

    d = pd.DataFrame([r.__dict__ for r in results]).replace([np.inf, -np.inf], np.nan).dropna(subset=["iq", "I"])
    d = d.reset_index(drop=True)

    print()
    print("usable_subjects", int(d.shape[0]))
    if skipped:
        print("skipped", skipped)

    # Validate: IQ ~ I (equation output), report prediction error
    X = d[["I"]].to_numpy()
    y = d["iq"].to_numpy()

    loo = LeaveOneOut()
    model = LinearRegression()
    yhat = cross_val_predict(model, X, y, cv=loo)

    mae = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))

    base = np.full_like(y, y.mean(), dtype=float)
    base_mae = mean_absolute_error(y, base)
    base_rmse = float(np.sqrt(mean_squared_error(y, base)))

    corr = float(np.corrcoef(y, yhat)[0, 1]) if y.size >= 2 else float("nan")
    corr_raw = float(np.corrcoef(d["I"].to_numpy(), y)[0, 1]) if y.size >= 2 else float("nan")

    print()
    print("IQ mean/std", float(y.mean()), float(y.std(ddof=1)) if y.size > 1 else 0.0)
    print("I  mean/std", float(d["I"].mean()), float(d["I"].std(ddof=1)) if d.shape[0] > 1 else 0.0)
    print("LOO  IQ~I  MAE", float(mae), "RMSE", float(rmse), "corr(y,yhat)", float(corr))
    print("BASELINE    MAE", float(base_mae), "RMSE", float(base_rmse))
    print("raw_corr(I, IQ)", float(corr_raw))

    # Worst errors for quick inspection
    err = np.abs(y - yhat)
    worst_idx = np.argsort(-err)[: min(10, err.size)]
    worst = d.loc[worst_idx, ["subject", "iq", "I"]].copy()
    worst["pred_iq"] = yhat[worst_idx]
    worst["abs_err"] = err[worst_idx]

    print()
    print("worst_abs_errors (top 10)")
    print(worst.sort_values("abs_err", ascending=False).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

