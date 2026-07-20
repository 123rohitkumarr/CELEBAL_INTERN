import re

try:
    import cohere
except ImportError:  # pragma: no cover - import guard for environments without the package
    cohere = None

import fitz

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover - import guard for environments without the package
    Pinecone = None
    ServerlessSpec = None


class VectorStore:
    def __init__(self, pdf_path: str, cohere_api_key: str, pinecone_api_key: str):
        self.pdf_path = pdf_path
        self.co = None
        if cohere_api_key and cohere is not None:
            try:
                self.co = cohere.Client(cohere_api_key)
            except Exception:
                self.co = None

        self.pinecone_api_key = pinecone_api_key
        self.chunks = []
        self.embeddings = []
        self.retrieve_top_k = 5
        self.rerank_top_k = 3
        self.index = None
        self.load_pdf()
        self.split_text()
        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text.strip()

    def split_text(self, chunk_size=700):
        if not getattr(self, "pdf_text", ""):
            self.chunks = []
            return

        sentences = re.split(r"(?<=[.!?])\s+", self.pdf_text)
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    self.chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            self.chunks.append(current_chunk)

    def embed_chunks(self, batch_size=90):
        if not self.chunks or self.co is None:
            return

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            try:
                batch_embeddings = self.co.embed(
                    texts=batch,
                    input_type="search_document",
                    model="embed-english-v3.0",
                ).embeddings
                self.embeddings.extend(batch_embeddings)
            except Exception:
                self.embeddings = []
                break

    def index_chunks(self):
        if not self.chunks or self.pinecone_api_key is None or Pinecone is None:
            self.index = None
            return

        if self.embeddings:
            try:
                pc = Pinecone(api_key=self.pinecone_api_key)
                index_name = "rag-qa-bot"
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=len(self.embeddings[0]),
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                self.index = pc.Index(index_name)
                chunks_metadata = [{"text": chunk} for chunk in self.chunks]
                ids = [str(i) for i in range(len(self.chunks))]
                self.index.upsert(vectors=list(zip(ids, self.embeddings, chunks_metadata)))
            except Exception:
                self.index = None

    def retrieve(self, query: str) -> list:
        if not self.chunks:
            return []

        if self.index is not None and self.co is not None:
            try:
                query_emb = self.co.embed(
                    texts=[query],
                    model="embed-english-v3.0",
                    input_type="search_query",
                ).embeddings
                res = self.index.query(vector=query_emb, top_k=self.retrieve_top_k, include_metadata=True)
                docs_to_rerank = [match["metadata"]["text"] for match in res["matches"]]
                rerank_results = self.co.rerank(
                    query=query,
                    documents=docs_to_rerank,
                    top_n=self.rerank_top_k,
                    model="rerank-english-v2.0",
                )
                return [res["matches"][result.index]["metadata"] for result in rerank_results.results]
            except Exception:
                pass

        query_words = {word.lower() for word in re.findall(r"[a-zA-Z0-9]+", query)}
        scored_chunks = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [{"text": chunk} for _, chunk in scored_chunks[: self.retrieve_top_k]]
