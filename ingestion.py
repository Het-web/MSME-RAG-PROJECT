from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from utils import sha256_file, sha256_text, load_json, save_json


class PDFIngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )
        self.vectorstore = Chroma(
            collection_name=self.settings.chroma_collection_name,
            persist_directory=str(self.settings.chroma_persist_dir),
            embedding_function=self.embeddings,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_all(self) -> dict[str, int]:
        """
        Ingest all PDFs from the configured PDF directory into ChromaDB.

        This version writes documents in small batches to avoid large
        embed_documents() calls, which can cause Ollama runner failures.
        """
        manifest = load_json(
            self.settings.ingestion_manifest_path,
            {"files": {}, "chunk_hashes": []},
        )
        known_chunk_hashes = set(manifest.get("chunk_hashes", []))
        stats = {"pdfs_seen": 0, "pdfs_skipped": 0, "chunks_added": 0}

        # Small batches reduce the chance of Ollama/LangChain runner issues.
        batch_size = 10

        for pdf_path in sorted(self.settings.pdf_dir.glob("*.pdf")):
            stats["pdfs_seen"] += 1
            file_hash = sha256_file(pdf_path)
            manifest_file = manifest.get("files", {}).get(str(pdf_path))

            # Skip unchanged files.
            if manifest_file and manifest_file.get("file_hash") == file_hash:
                stats["pdfs_skipped"] += 1
                continue

            chunks = self._load_and_chunk(pdf_path)
            new_docs: list[Document] = []

            for index, doc in enumerate(chunks):
                chunk_hash = sha256_text(doc.page_content)
                if chunk_hash in known_chunk_hashes:
                    continue

                chunk_id = f"{pdf_path.stem}-{index}-{chunk_hash[:12]}"
                doc.metadata.update(
                    {
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "document_type": "pdf",
                        "upload_date": datetime.fromtimestamp(
                            pdf_path.stat().st_mtime, timezone.utc
                        ).isoformat(),
                        "chunk_id": chunk_id,
                        "chunk_hash": chunk_hash,
                    }
                )
                new_docs.append(doc)
                known_chunk_hashes.add(chunk_hash)

            # Add documents in batches instead of one huge call.
            if new_docs:
                for start in range(0, len(new_docs), batch_size):
                    batch = new_docs[start : start + batch_size]
                    self.vectorstore.add_documents(
                        batch,
                        ids=[doc.metadata["chunk_id"] for doc in batch],
                    )
                    stats["chunks_added"] += len(batch)

            manifest.setdefault("files", {})[str(pdf_path)] = {
                "file_hash": file_hash,
                "last_ingested_at": datetime.now(timezone.utc).isoformat(),
                "chunks": len(chunks),
            }

            # Save after each PDF so progress is not lost if one later file fails.
            manifest["chunk_hashes"] = sorted(known_chunk_hashes)
            save_json(self.settings.ingestion_manifest_path, manifest)

        # Final save.
        manifest["chunk_hashes"] = sorted(known_chunk_hashes)
        save_json(self.settings.ingestion_manifest_path, manifest)
        return stats

    def _load_and_chunk(self, pdf_path: Path) -> list[Document]:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["page"] = page.metadata.get("page")
        return self.splitter.split_documents(pages)