import re

try:
    from groq import Groq
except ImportError:  # pragma: no cover - import guard
    Groq = None

from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and Groq is not None else None

SYSTEM_PROMPT = """
You are an intent classifier.

Classify the user's query into EXACTLY one of these intents:

1. MATH
2. KEYWORD
3. GENERAL

Rules:

- MATH:
Any arithmetic, calculations,
word problems,
percentages,
equations,
numeric reasoning.

- KEYWORD:
User wants important words,
keywords,
tags,
main concepts.

- GENERAL:
Everything else.

Return ONLY one word.
"""


def _fallback_classify(query: str) -> str:
    text = query.lower().strip()
    if any(word in text for word in ["keyword", "keywords", "extract keywords", "important words"]):
        return "KEYWORD"
    if any(word in text for word in ["calculate", "solve", "sum", "plus", "minus", "multiply", "divide", "+", "-", "*", "/"]):
        return "MATH"
    if re.search(r"\d", text):
        return "MATH"
    return "GENERAL"


def classify_intent(query: str):
    if client is None:
        return _fallback_classify(query)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _fallback_classify(query)