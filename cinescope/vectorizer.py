"""A small, serializable TF-IDF vectorizer."""

from __future__ import annotations

import math
from collections import Counter

from .preprocess import tokenize


class TfidfVectorizer:
    def __init__(
        self,
        max_features: int = 12000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> None:
        if max_features < 1 or min_df < 1:
            raise ValueError("max_features and min_df must be positive")
        if len(ngram_range) != 2 or not 1 <= ngram_range[0] <= ngram_range[1]:
            raise ValueError("ngram_range must be an ordered pair of positive integers")
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        features: list[str] = []
        for size in range(self.ngram_range[0], self.ngram_range[1] + 1):
            features.extend(
                " ".join(tokens[index:index + size])
                for index in range(len(tokens) - size + 1)
            )
        return features

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()
        for text in texts:
            features = self._features(text)
            term_frequency.update(features)
            document_frequency.update(set(features))
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
            counts = Counter(self._features(text))
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
            "ngram_range": list(self.ngram_range),
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfVectorizer":
        instance = cls(
            int(data["max_features"]),
            int(data["min_df"]),
            tuple(data.get("ngram_range", [1, 1])),
        )
        instance.vocabulary = {str(k): int(v) for k, v in data["vocabulary"].items()}
        instance.idf = [float(value) for value in data["idf"]]
        return instance
