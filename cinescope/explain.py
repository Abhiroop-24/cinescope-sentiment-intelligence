"""Human-readable feature contributions for a prediction."""

from .preprocess import tokenize
from .vectorizer import TfidfVectorizer
from .model import LogisticRegression


def explain(text: str, vectorizer: TfidfVectorizer, model: LogisticRegression, limit: int = 8) -> list[dict]:
    tokens = tokenize(text)
    contributions: dict[str, float] = {}
    for token in tokens:
        index = vectorizer.vocabulary.get(token)
        if index is not None:
            contributions[token] = contributions.get(token, 0.0) + model.weights[index] * vectorizer.idf[index]
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
    return [{"term": term, "contribution": round(value, 5), "direction": "positive" if value >= 0 else "negative"}
            for term, value in ranked]
