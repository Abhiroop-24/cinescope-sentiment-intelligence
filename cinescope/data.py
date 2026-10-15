"""Official IMDb dataset download and loading."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(DATA_URL, archive)
    extracted = data_dir / "aclImdb"
    if not extracted.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
    return extracted


def load_split(root: Path, split: str, limit: int | None = None) -> tuple[list[str], list[int]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    texts: list[str] = []
    labels: list[int] = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        files = sorted((root / split / label_name).glob("*.txt"))
        for path in files[:limit] if limit else files:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
            labels.append(label)
    if not texts:
        raise FileNotFoundError(f"no IMDb reviews found under {root / split}")
    return texts, labels
