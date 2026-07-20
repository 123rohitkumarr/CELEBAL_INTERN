import json
import re

try:
    from groq import Groq
except ImportError:  # pragma: no cover - import guard
    Groq = None

from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and Groq is not None else None

SYSTEM_PROMPT = """
You are an AI planner.

You have access to the following tools:

1. calculator
Description:
Performs mathematical calculations.

Arguments:
{
    "expression": "2+3*5"
}

------------------------------------------------

2. keyword_tool

Description:
Extracts important keywords from text.

Arguments:
{
    "text": "Artificial Intelligence is changing healthcare."
}

------------------------------------------------

If no tool is required, return:

{
    "tool": "GENERAL",
    "arguments": {}
}

Return ONLY valid JSON.
"""


def _fallback_plan(query: str):
    text = query.lower().strip()
    if any(word in text for word in ["keyword", "keywords", "extract keywords", "important words"]):
        return {"tool": "keyword_tool", "arguments": {"text": query}}
    if any(word in text for word in ["calculate", "solve", "sum", "plus", "minus", "multiply", "divide"]):
        expression = re.sub(r"[^0-9+\-*/(). ]", "", query)
        return {"tool": "calculator", "arguments": {"expression": expression.strip() or query}}
    return {"tool": "GENERAL", "arguments": {}}


def create_plan(query: str):
    if client is None:
        return _fallback_plan(query)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return _fallback_plan(query)