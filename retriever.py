import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import get_settings
from models import RetrievalResult, Source
from utils import compact_text, elapsed_ms, now_ms


MSME_DEFINITION_QUERY_PATTERN = re.compile(
    r"\b(definition|define|classification|classify|thresholds?|limits?|categories|what is (?:an? )?msme|what are msmes?)\b",
    re.IGNORECASE,
)


class MSMERetriever:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            persist_directory=str(settings.chroma_persist_dir),
            embedding_function=self.embeddings,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        metrics: dict[str, float | int] = {}

        search_query = query
        if MSME_DEFINITION_QUERY_PATTERN.search(query) and "msme" in query.lower():
            search_query = f"{query} classification criteria investment turnover limits threshold micro small medium"

        embedding_start = now_ms()
        query_embedding = self.embeddings.embed_query(search_query)
        metrics["embedding_generation_ms"] = elapsed_ms(embedding_start)

        retrieval_start = now_ms()
        docs = self.vectorstore.similarity_search_by_vector(
            query_embedding,
            k=self.settings.retrieval_top_k,
        )
        metrics["chroma_retrieval_ms"] = elapsed_ms(retrieval_start)
        metrics["retrieved_chunks"] = len(docs)

        sources = [
            Source(
                source_file=doc.metadata.get("source_file"),
                document_type=doc.metadata.get("document_type"),
                upload_date=doc.metadata.get("upload_date"),
                chunk_id=doc.metadata.get("chunk_id"),
                page=doc.metadata.get("page"),
                content_preview=compact_text(doc.page_content, 240),
            )
            for doc in docs
        ]
        context = "\n\n".join(
            f"[{index + 1}] Source: {source.source_file or 'unknown'}; Chunk: {source.chunk_id or 'unknown'}\n{doc.page_content}"
            for index, (doc, source) in enumerate(zip(docs, sources, strict=False))
        )
        return RetrievalResult(context=context, sources=sources, metrics=metrics)
