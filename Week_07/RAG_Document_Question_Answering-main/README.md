# RAG-Based Document Question Answering System 🤖📄

This project now provides a more reliable document Q&A flow for PDF files. It can work with external Cohere and Pinecone services when API keys are provided, and it also falls back to local retrieval so the experience remains useful even without those services.

## What changed

- Improved the Streamlit experience for uploading PDFs and asking questions.
- Fixed the retrieval pipeline so document chunks are processed consistently.
- Added a safe fallback answer path when API-based services are unavailable.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
cd src
streamlit run app.py
```

4. Upload a PDF document and ask questions.

## Project structure

```bash
src/
  app.py
  chatbot.py
  vectorstore.py
```

## Notes

- Cohere and Pinecone keys are optional for local demo use.
- If keys are missing, the app still responds using the most relevant text chunks from the uploaded document.