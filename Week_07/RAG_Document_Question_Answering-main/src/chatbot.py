import uuid

try:
    import cohere
except ImportError:  # pragma: no cover - import guard for environments without the package
    cohere = None


class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = None
        if cohere_api_key and cohere is not None:
            try:
                self.co = cohere.Client(cohere_api_key)
            except Exception:
                self.co = None

    def _fallback_response(self, user_message: str, retrieved_docs: list):
        if not retrieved_docs:
            return "I could not find enough relevant content in the uploaded document to answer this question."

        top_doc = retrieved_docs[0]
        if isinstance(top_doc, dict) and "text" in top_doc:
            context = top_doc["text"]
        else:
            context = str(top_doc)

        return (
            f"Based on the document, the closest match is: {context[:400]}"
            if len(context) > 400
            else f"Based on the document, the closest match is: {context}"
        )

    def respond(self, user_message: str):
        retrieved_docs = self.vectorstore.retrieve(user_message)

        if self.co is None:
            return self._fallback_response(user_message, retrieved_docs), retrieved_docs

        try:
            response = self.co.chat(
                message=user_message,
                model="command-r",
                search_queries_only=True,
            )
            if getattr(response, "search_queries", None):
                docs = []
                for query in response.search_queries:
                    docs.extend(self.vectorstore.retrieve(query.text))
                response = self.co.chat_stream(
                    message=user_message,
                    model="command-r",
                    documents=docs,
                    conversation_id=self.conversation_id,
                )
                return response, docs

            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                conversation_id=self.conversation_id,
            )
            return response, retrieved_docs
        except Exception:
            return self._fallback_response(user_message, retrieved_docs), retrieved_docs
