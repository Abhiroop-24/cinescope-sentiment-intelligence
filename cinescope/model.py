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
