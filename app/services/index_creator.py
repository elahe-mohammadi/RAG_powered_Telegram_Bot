from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class VSSettings(BaseSettings):
    data_path: str = "app/data"
    index_path: str = "app/faiss_index"
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
    env_prefix="RAG_INDEX_",
    env_file=".env",
    extra="ignore",
    )

class VectorStore:
    def __init__(self, settings: VSSettings):
        self.settings = settings
        # Initialize our embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        # Load existing FAISS index or build a new one
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> FAISS:
        index_dir = Path(self.settings.index_path)
        faiss_file = index_dir / "index.faiss"
        if faiss_file.exists():
            try:
                return FAISS.load_local(str(index_dir), self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                pass  # fall through to rebuild on any load error
        
        documents = self._load_and_split_documents()
        faiss_index = FAISS.from_documents(documents, self.embeddings)
        faiss_index.save_local(str(index_dir))
        return faiss_index

    def _load_and_split_documents(self) -> List[Document]:
        # Discover all .txt files in our data directory
        txt_files = list(Path(self.settings.data_path).glob("*.txt"))
        raw_docs: List[Document] = []
        for file in txt_files:
            loader = TextLoader(file_path=str(file),
                                encoding="utf-8")
            raw_docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        return splitter.split_documents(raw_docs)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.index.similarity_search(query, k=k)
