import os
import tempfile

import streamlit as st

from chatbot import Chatbot
from vectorstore import VectorStore


def _extract_text(response):
    if isinstance(response, str):
        return response
    if not response:
        return ""

    parts = []
    for event in response:
        event_text = getattr(event, "text", "")
        if event_text:
            parts.append(event_text)
    return "".join(parts)


def main():
    st.set_page_config(page_title="Document QA Bot", page_icon="🤖", layout="wide")
    st.title("Document QA Bot 🤖")
    st.caption("Upload a PDF and ask questions about its content.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

    with st.sidebar:
        st.header("API Keys 🔑")
        cohere_api_key = st.text_input("Cohere API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")

        if st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.session_state["vectorstore"] = None
            st.success("Chat history cleared.")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    user_query = st.text_input("Ask a question based on the document")

    if st.button("Process document") and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                vectorstore = VectorStore(temp_path, cohere_api_key, pinecone_api_key)
                st.session_state["vectorstore"] = vectorstore
                st.success("Document processed successfully.")
            except Exception as exc:
                st.error(f"Unable to process the PDF: {exc}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.button("Ask") and user_query.strip():
        if st.session_state["vectorstore"] is None:
            st.error("Please process a PDF document before asking a question.")
        else:
            with st.spinner("Generating a response..."):
                chatbot = Chatbot(st.session_state["vectorstore"], cohere_api_key)
                response, retrieved_docs = chatbot.respond(user_query)
                answer = _extract_text(response)
                st.session_state["chat_history"].append((user_query, answer, retrieved_docs))

    if st.session_state["chat_history"]:
        st.subheader("Conversation")
        for user_query, response, retrieved_docs in st.session_state["chat_history"]:
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**Bot:** {response}")
            if retrieved_docs:
                with st.expander("Relevant snippets"):
                    for doc in retrieved_docs[:3]:
                        if isinstance(doc, dict) and "text" in doc:
                            st.write(doc["text"])
                        else:
                            st.write(doc)


if __name__ == "__main__":
    main()
