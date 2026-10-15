"""Text normalization shared by training and inference."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in "
    "into is it its me my of on or our she so than that the their them there "
    "they this to was we were what when which who will with you your".split()
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def normalize(text: str, *, remove_stopwords: bool = True) -> str:
    return " ".join(tokenize(text, remove_stopwords=remove_stopwords))


def token_counts(texts: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    return counts
