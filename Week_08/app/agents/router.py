import re

MATH_PATTERN = re.compile(r"[0-9+\-*/()%]+")

KEYWORD_WORDS = [
    "keyword",
    "keywords",
    "extract keywords",
    "important words",
]


def detect_intent(query: str) -> str:

    query = query.lower().strip()

    # keyword extraction

    for word in KEYWORD_WORDS:
        if word in query:
            return "KEYWORD"

    # math

    if "calculate" in query:
        return "MATH"

    if "solve" in query:
        return "MATH"

    if MATH_PATTERN.search(query):
        return "MATH"

    return "GENERAL"