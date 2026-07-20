import string

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover - import guard
    stopwords = None
    word_tokenize = None


def _get_stop_words():
    if stopwords is None:
        return set()
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords")
            return set(stopwords.words("english"))
        except Exception:
            return set()


STOP_WORDS = _get_stop_words()


def extract_keywords(text: str):
    try:
        if word_tokenize is None:
            tokens = text.lower().replace("\n", " ").split()
        else:
            tokens = word_tokenize(text.lower())

        keywords = []
        for token in tokens:
            if (
                token not in STOP_WORDS
                and token not in string.punctuation
                and token.isalpha()
            ):
                keywords.append(token)

        keywords = list(dict.fromkeys(keywords))
        return {"success": True, "keywords": keywords}

    except Exception as exc:
        return {"success": False, "error": str(exc)}