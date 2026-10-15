"""Optional reproduction comparison using scikit-learn.

This command is deliberately explicit about its dependency and produces a
JSON result file rather than embedding paper numbers in the application.
"""

from __future__ import annotations

import json
from pathlib import Path


def run(data_root: Path, output: Path, limit: int | None = None) -> dict:
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import LinearSVC
    except ImportError as error:
        raise RuntimeError(
            "Full comparison requires optional dependencies. Install requirements-optional.txt."
        ) from error
    from .data import load_split

    train_text, y_train = load_split(data_root, "train", limit)
    test_text, y_test = load_split(data_root, "test", limit)
    vectorizers = {
        "binary_3gram": CountVectorizer(binary=True, ngram_range=(1, 3), max_features=50000),
        "count_3gram": CountVectorizer(ngram_range=(1, 3), max_features=50000),
        "tfidf_3gram": TfidfVectorizer(ngram_range=(1, 3), max_features=50000),
    }
    classifiers = {
        "logistic_regression": LogisticRegression(C=1, max_iter=300, random_state=42),
        "linear_svm": LinearSVC(C=1, random_state=42),
        "naive_bayes": MultinomialNB(alpha=1),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "dnn": MLPClassifier(hidden_layer_sizes=(30, 30, 20, 10, 10), max_iter=30,
                             early_stopping=True, random_state=42),
    }
    results = []
    for representation, vectorizer in vectorizers.items():
        train_matrix = vectorizer.fit_transform(train_text)
        test_matrix = vectorizer.transform(test_text)
        for name, classifier in classifiers.items():
            classifier.fit(train_matrix, y_train)
            predictions = classifier.predict(test_matrix)
            results.append({
                "model": name,
                "representation": representation,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_positive": precision_score(y_test, predictions, zero_division=0),
                "recall_positive": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return {"rows": len(results), "output": str(output)}
