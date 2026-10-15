"""JSON persistence for model and preprocessing artifacts."""

import json
from pathlib import Path

from .model import LogisticRegression
from .vectorizer import TfidfVectorizer


def save_artifact(path: Path, vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vectorizer": vectorizer.to_dict(), "model": model.to_dict()}), encoding="utf-8")


def load_artifact(path: Path) -> tuple[TfidfVectorizer, LogisticRegression]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TfidfVectorizer.from_dict(data["vectorizer"]), LogisticRegression.from_dict(data["model"])
