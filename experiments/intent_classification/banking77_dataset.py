import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from urllib.request import urlretrieve


BANKING77_URLS = {
    "train": "https://github.com/PolyAI-LDN/task-specific-datasets/raw/master/banking_data/train.csv",
    "test": "https://github.com/PolyAI-LDN/task-specific-datasets/raw/master/banking_data/test.csv",
}


def _download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    urlretrieve(url, str(dest_path))


def ensure_banking77_downloaded(root_dir: Path) -> None:
    root_dir = Path(root_dir)
    for split, url in BANKING77_URLS.items():
        target = root_dir / f"{split}.csv"
        _download_file(url, target)


def _read_csv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_header = header is not None and "text" in header[0].lower()
        if has_header:
            for row in reader:
                if len(row) < 2:
                    continue
                text, label = row[0], row[1]
                rows.append((text, label))
        else:
            if header:
                if len(header) >= 2:
                    rows.append((header[0], header[1]))
            for row in reader:
                if len(row) < 2:
                    continue
                rows.append((row[0], row[1]))
    return rows


def build_label_mapping(train_csv: Path) -> Dict[str, int]:
    pairs = _read_csv(train_csv)
    labels = sorted({lbl for _, lbl in pairs})
    return {lbl: i for i, lbl in enumerate(labels)}


class Banking77Dataset(Dataset):
    """
    Banking77 intent classification dataset.

    Each item:
        {
            "text": str,
            "label": int,
        }
    """

    def __init__(self, root_dir: str, split: str = "train") -> None:
        self.root_dir = Path(root_dir)
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}")

        ensure_banking77_downloaded(self.root_dir)

        train_csv = self.root_dir / "train.csv"
        split_csv = self.root_dir / f"{split}.csv"

        if not train_csv.exists() or not split_csv.exists():
            raise FileNotFoundError("Banking77 CSV files not found after download.")

        self.label2id = build_label_mapping(train_csv)

        pairs = _read_csv(split_csv)
        self.texts: List[str] = []
        self.labels: List[int] = []
        for text, label_str in pairs:
            if label_str not in self.label2id:
                continue
            self.texts.append(text)
            self.labels.append(self.label2id[label_str])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
        }


