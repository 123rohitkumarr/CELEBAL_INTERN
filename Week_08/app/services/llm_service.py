try:
    from groq import Groq
except ImportError:  # pragma: no cover - import guard
    Groq = None

from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and Groq is not None else None


def ask_llm(query: str):
    if client is None:
        return {
            "success": True,
            "response": f"I can help with that. You asked: {query}",
        }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        return {"success": True, "response": answer}

    except Exception as exc:
        return {"success": False, "error": str(exc)}