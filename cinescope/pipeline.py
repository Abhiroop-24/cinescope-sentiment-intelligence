"""Train/evaluate/predict orchestration."""

from pathlib import Path
from .data import load_split
from .metrics import classification_metrics
from .model import LogisticRegression, fit_torch
from .persistence import load_artifact, save_artifact
from .vectorizer import TfidfVectorizer


def train(data_root: Path, artifact: Path, max_features: int = 12000, limit: int | None = None) -> dict:
    texts, labels = load_split(data_root, "train", limit)
    vectorizer = TfidfVectorizer(max_features=max_features)
    rows = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(rows, labels, len(vectorizer.vocabulary))
    save_artifact(artifact, vectorizer, model)
    return {"samples": len(labels), "features": len(vectorizer.vocabulary), "artifact": str(artifact)}


def train_accelerated(
    data_root: Path,
    artifact: Path,
    *,
    max_features: int = 12000,
    limit: int | None = None,
    device: str = "auto",
) -> dict:
    texts, labels = load_split(data_root, "train", limit)
    vectorizer = TfidfVectorizer(max_features=max_features)
    rows = vectorizer.fit_transform(texts)
    model = fit_torch(rows, labels, len(vectorizer.vocabulary), device=device)
    save_artifact(artifact, vectorizer, model)
    return {
        "samples": len(labels),
        "features": len(vectorizer.vocabulary),
        "device": device,
        "artifact": str(artifact),
    }


def evaluate(data_root: Path, artifact: Path, limit: int | None = None) -> dict:
    texts, labels = load_split(data_root, "test", limit)
    vectorizer, model = load_artifact(artifact)
    predictions = [model.predict_one(row) for row in vectorizer.transform(texts)]
    return classification_metrics(labels, predictions)
