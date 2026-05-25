# AZURE SETUP REQUIRED:
# 1. Create an Azure AI Search service (Basic tier or above)
# 2. Note the service endpoint and admin key from the Azure portal
# 3. Set env vars: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME
# 4. Set Azure OpenAI env vars for embedding: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#    AZURE_OPENAI_EMBEDDING_DEPLOYMENT (text-embedding-3-small), AZURE_OPENAI_API_VERSION
# LOCAL FALLBACK: If Azure Search is not configured, a FAISS in-memory index is used
#                 with the same search interface. Embeddings still require Azure OpenAI
#                 OR fall back to a mock embedding for pure local testing.

# stdlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Unified vector store that supports Azure AI Search (production) and
    FAISS (local development fallback) with an identical search interface.
    """

    def __init__(self) -> None:
        """Initialise lazily — connections are made on first method call."""
        self._azure_search_client: Any = None
        self._faiss_index: Any = None
        self._faiss_docs: List[Dict[str, Any]] = []
        self._embeddings_client: Any = None
        self._use_azure: bool = False
        self._initialized: bool = False

    def _get_embeddings(self) -> Any:
        """
        Return an embeddings client (Azure OpenAI or mock).

        Returns:
            AzureOpenAIEmbeddings instance or a mock object.
        """
        if self._embeddings_client is not None:
            return self._embeddings_client

        from langchain_huggingface import HuggingFaceEmbeddings

        self._embeddings_client = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        return self._embeddings_client

    def _init_azure_search(self) -> bool:
        """
        Initialise Azure AI Search client.

        Returns:
            True if successfully connected, False otherwise.
        """
        from config import settings

        if not settings.azure_search_configured:
            return False

        try:
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
            from azure.core.credentials import AzureKeyCredential

            SEARCH_ENDPOINT = settings.azure_search_endpoint
            SEARCH_KEY = settings.azure_search_key
            INDEX_NAME = settings.azure_search_index_name

            self._azure_search_client = SearchClient(
                endpoint=SEARCH_ENDPOINT,
                index_name=INDEX_NAME,
                credential=AzureKeyCredential(SEARCH_KEY)
            )
            logger.info(
                "Azure AI Search client connected to index: %s",
                INDEX_NAME,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to connect to Azure AI Search: %s", exc)
            return False

    def _init_faiss(self) -> None:
        """Initialise a FAISS in-memory vector index as local fallback."""
        try:
            import faiss
            import numpy as np

            self._faiss_dimension = 384  # text-embedding-3-small dimensions
            self._faiss_index = faiss.IndexFlatIP(self._faiss_dimension)  # inner product
            logger.info("FAISS in-memory vector index initialised (dim=%d).", self._faiss_dimension)
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            self._faiss_index = None

    def _ensure_initialized(self) -> None:
        """Ensure the vector store backend is initialised (called lazily)."""
        if self._initialized:
            return
        self._use_azure = self._init_azure_search()
        if not self._use_azure:
            logger.info("Falling back to FAISS local vector store.")
            self._init_faiss()
        self._initialized = True

    def create_index_if_not_exists(self) -> None:
        """
        Create the Azure AI Search index schema if it does not already exist.
        """
        from config import settings

        if not settings.azure_search_configured:
            logger.info("Azure Search not configured — skipping index creation.")
            return

        try:
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                SearchIndex,
                SimpleField,
                SearchableField,
                SearchField,
                SearchFieldDataType,
                VectorSearch,
                HnswAlgorithmConfiguration,
                VectorSearchProfile,
            )
            from azure.core.credentials import AzureKeyCredential

            SEARCH_ENDPOINT = settings.azure_search_endpoint
            SEARCH_KEY = settings.azure_search_key
            INDEX_NAME = settings.azure_search_index_name

            index_client = SearchIndexClient(
                endpoint=SEARCH_ENDPOINT,
                credential=AzureKeyCredential(SEARCH_KEY)
            )
            
            try:
                index_client.delete_index(INDEX_NAME)
                print(f"Deleted old index {INDEX_NAME}")
            except Exception:
                print("No existing index to delete")
                
            try:
                index_client.get_index(INDEX_NAME)
                print(f"Index {INDEX_NAME} already exists")
            except Exception:
                fields = [
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                    SearchableField(name="content", type=SearchFieldDataType.String),
                    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
                    SearchField(
                        name="embedding",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=384,
                        vector_search_profile_name="hnsw-profile"
                    ),
                ]
                vector_search = VectorSearch(
                    algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
                    profiles=[VectorSearchProfile(
                        name="hnsw-profile",
                        algorithm_configuration_name="hnsw-algo"
                    )]
                )
                index = SearchIndex(
                    name=INDEX_NAME, fields=fields, vector_search=vector_search
                )
                index_client.create_index(index)
                print(f"Created index {INDEX_NAME}")

        except Exception as exc:
            logger.error("Failed to create Azure AI Search index: %s", exc)
            raise

    def ingest_pdfs(self, pdf_folder: str) -> int:
        """
        Load all PDFs in the given folder, chunk them, embed them, and upload
        to the vector store (Azure AI Search or FAISS).

        Args:
            pdf_folder: Path to the folder containing PDF files.

        Returns:
            Total number of document chunks ingested.
        """
        self._ensure_initialized()
        pdf_path = Path(pdf_folder)
        pdf_files = list(pdf_path.glob("*.txt"))

        if not pdf_files:
            logger.warning("No txt files found in: %s", pdf_folder)
            return 0

        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )

        embeddings = self._get_embeddings()
        total_chunks = 0

        for pdf_file in pdf_files:
            logger.info("Loading TXT: %s", pdf_file.name)
            try:
                loader = TextLoader(str(pdf_file), encoding="utf-8")
                pages = loader.load()
                chunks = splitter.split_documents(pages)

                texts = [c.page_content for c in chunks]
                vectors = embeddings.embed_documents(texts)

                if self._use_azure:
                    documents = [
                        {
                            "id": str(uuid.uuid4()),
                            "content": texts[i],
                            "source": pdf_file.name,
                            "embedding": vectors[i],
                        }
                        for i in range(len(texts))
                    ]
                    self._azure_search_client.upload_documents(documents=documents)
                    logger.info(
                        "Uploaded %d chunks from %s to Azure AI Search.",
                        len(documents),
                        pdf_file.name,
                    )
                else:
                    # FAISS fallback
                    import numpy as np

                    if self._faiss_index is not None:
                        vecs = np.array(vectors, dtype="float32")
                        # Normalise for cosine similarity via inner product
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        vecs = vecs / (norms + 1e-10)
                        self._faiss_index.add(vecs)

                    for i, text in enumerate(texts):
                        self._faiss_docs.append(
                            {"content": text, "source": pdf_file.name}
                        )

                total_chunks += len(texts)

            except Exception as exc:
                logger.error("Error ingesting PDF %s: %s", pdf_file.name, exc)

        logger.info("PDF ingestion complete. Total chunks: %d", total_chunks)
        return total_chunks

    def search_documents(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search for the given query.

        Args:
            query: Natural language search query.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: content, source, score.
        """
        self._ensure_initialized()
        embeddings = self._get_embeddings()
        query_vector = embeddings.embed_query(query)

        if self._use_azure:
            return self._search_azure(query, query_vector, top_k)
        return self._search_faiss(query_vector, top_k)

    def _search_azure(
        self, query: str, query_vector: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Query Azure AI Search using vector similarity.

        Args:
            query: Natural language query string.
            query_vector: Embedded query vector.
            top_k: Number of results to return.

        Returns:
            List of result dicts.
        """
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        results = self._azure_search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["content", "source"],
            top=top_k,
        )

        return [
            {
                "content": r["content"],
                "source": r["source"],
                "score": r.get("@search.score", 0.0),
            }
            for r in results
        ]

    def _search_faiss(
        self, query_vector: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Query FAISS in-memory index using inner-product similarity.

        Args:
            query_vector: Embedded query vector.
            top_k: Number of results to return.

        Returns:
            List of result dicts.
        """
        if self._faiss_index is None or len(self._faiss_docs) == 0:
            logger.warning("FAISS index is empty — no documents have been ingested.")
            return []

        import numpy as np

        vec = np.array([query_vector], dtype="float32")
        norm = np.linalg.norm(vec)
        vec = vec / (norm + 1e-10)

        actual_k = min(top_k, len(self._faiss_docs))
        scores, indices = self._faiss_index.search(vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_docs):
                continue
            doc = self._faiss_docs[idx]
            results.append(
                {
                    "content": doc["content"],
                    "source": doc["source"],
                    "score": float(score),
                }
            )
        return results


# Module-level singleton
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the module-level singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vs = get_vector_store()
    vs.create_index_if_not_exists()
    vs.ingest_pdfs("data")
