"""Sparse logistic regression trained with deterministic SGD."""

from __future__ import annotations

import math
import random


class LogisticRegression:
    def __init__(self, epochs: int = 12, learning_rate: float = 0.35, l2: float = 1e-4) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-60.0, min(60.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def fit(self, rows: list[dict[int, float]], labels: list[int], features: int) -> "LogisticRegression":
        if len(rows) != len(labels) or not rows:
            raise ValueError("rows and labels must be non-empty and have equal length")
        self.weights = [0.0] * features
        order = list(range(len(rows)))
        rng = random.Random(42)
        for epoch in range(self.epochs):
            rng.shuffle(order)
            rate = self.learning_rate / (1.0 + epoch * 0.12)
            for row_index in order:
                row = rows[row_index]
                prediction = self._sigmoid(self.bias + sum(self.weights[i] * value for i, value in row.items()))
                error = prediction - labels[row_index]
                self.bias -= rate * error
                for index, value in row.items():
                    self.weights[index] -= rate * (error * value + self.l2 * self.weights[index])
        return self

    def predict_proba_one(self, row: dict[int, float]) -> float:
        if not self.weights:
            raise RuntimeError("model must be fitted before prediction")
        return self._sigmoid(self.bias + sum(self.weights[i] * value for i, value in row.items()))

    def predict_one(self, row: dict[int, float]) -> int:
        return int(self.predict_proba_one(row) >= 0.5)

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "learning_rate": self.learning_rate, "l2": self.l2,
                "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticRegression":
        instance = cls(int(data["epochs"]), float(data["learning_rate"]), float(data["l2"]))
        instance.weights = [float(value) for value in data["weights"]]
        instance.bias = float(data["bias"])
        return instance


def fit_torch(
    rows: list[dict[int, float]],
    labels: list[int],
    features: int,
    *,
    device: str = "auto",
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 0.08,
) -> LogisticRegression:
    """Train the same persisted model with batched PyTorch CUDA tensors."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU training requires PyTorch. Install a CUDA wheel with "
            "`pip install --index-url https://download.pytorch.org/whl/cu128 torch`."
        ) from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    target = torch.device(device)
    matrix = torch.zeros((len(rows), features), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for feature_index, value in row.items():
            matrix[row_index, feature_index] = value
    x = matrix.to(target)
    y = torch.tensor(labels, dtype=torch.float32, device=target).unsqueeze(1)
    weights = torch.zeros((features, 1), dtype=torch.float32, device=target, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, device=target, requires_grad=True)
    optimizer = torch.optim.Adam((weights, bias), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=target).manual_seed(42)
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator, device=target)
        for start in range(0, len(rows), batch_size):
            indices = order[start:start + batch_size]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                x.index_select(0, indices).matmul(weights) + bias,
                y.index_select(0, indices),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    result = LogisticRegression(epochs=epochs, learning_rate=learning_rate)
    result.weights = weights.detach().flatten().cpu().tolist()
    result.bias = float(bias.detach().cpu().item())
    return result
