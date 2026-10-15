"""A small, serializable TF-IDF vectorizer."""

from __future__ import annotations

import math
from collections import Counter

from .preprocess import tokenize


class TfidfVectorizer:
    def __init__(self, max_features: int = 12000, min_df: int = 2) -> None:
        if max_features < 1 or min_df < 1:
            raise ValueError("max_features and min_df must be positive")
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))
        candidates = [
            token for token, frequency in document_frequency.items()
            if frequency >= self.min_df
        ]
        candidates.sort(key=lambda token: (-term_frequency[token], token))
        terms = candidates[: self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        total_documents = max(len(texts), 1)
        self.idf = [
            math.log((1 + total_documents) / (1 + document_frequency[term])) + 1
            for term in terms
        ]
        return self

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        if not self.vocabulary:
            raise RuntimeError("vectorizer must be fitted before transform")
        rows: list[dict[int, float]] = []
        for text in texts:
            counts = Counter(tokenize(text))
            total = sum(counts.values()) or 1
            row: dict[int, float] = {}
            for token, count in counts.items():
                index = self.vocabulary.get(token)
                if index is not None:
                    row[index] = (count / total) * self.idf[index]
            norm = math.sqrt(sum(value * value for value in row.values())) or 1.0
            rows.append({index: value / norm for index, value in row.items()})
        return rows

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        return self.fit(texts).transform(texts)

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfVectorizer":
        instance = cls(int(data["max_features"]), int(data["min_df"]))
        instance.vocabulary = {str(k): int(v) for k, v in data["vocabulary"].items()}
        instance.idf = [float(value) for value in data["idf"]]
        return instance
