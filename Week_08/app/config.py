import os

from dotenv import load_dotenv

load_dotenv()


def get_groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()