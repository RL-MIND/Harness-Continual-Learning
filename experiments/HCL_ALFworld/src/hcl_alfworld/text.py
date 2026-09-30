from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "then",
    "to",
    "with",
    "you",
    "your",
    "this",
    "that",
    "there",
    "from",
    "into",
}


def tokenize(text: str) -> List[str]:
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOPWORDS]


def overlap_score(query: str, document: str) -> float:
    q, d = Counter(tokenize(query)), Counter(tokenize(document))
    if not q or not d:
        return 0.0
    common = sum(min(q[token], d[token]) for token in q)
    return common / max(1.0, (sum(q.values()) * sum(d.values())) ** 0.5)


def unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))
