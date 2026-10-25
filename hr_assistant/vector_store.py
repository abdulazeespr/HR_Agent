import os
import logging
from typing import Optional, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embedding_model
from hr_assistant.document_loader import load_hr_policy_documents
from hr_assistant.splitter import split_documents

logger = logging.getLogger(__name__)

def build_vector_store(documents: List[Document], embedding_model) -> FAISS:
    """
    Builds an in-memory FAISS vector store from chunked documents.
    """
    if not documents:
        raise ValueError("Cannot build vector store with empty documents.")
    try:
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,
        )
        logger.info(f"FAISS index built with {vector_store.index.ntotal} vectors.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise e

def save_vector_store(vector_store: FAISS, path: str = config.VECTOR_DB_PATH) -> bool:
    """
    Persists the FAISS vector store to disk.
    """
    try:
        os.makedirs(path, exist_ok=True)
        vector_store.save_local(path)
        logger.info(f"Vector store saved successfully to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save vector store to {path}: {e}")
        raise e

def vector_store_exists(path: str = config.VECTOR_DB_PATH) -> bool:
    """
    Checks if a valid FAISS index file exists at the specified path.
    """
    index_file = os.path.join(path, "index.faiss")
    pkl_file = os.path.join(path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)

def load_vector_store(embedding_model=None, path: str = config.VECTOR_DB_PATH) -> FAISS:
    """
    Loads an existing FAISS vector store from disk.
    """
    emb = embedding_model or get_embedding_model()
    if not vector_store_exists(path):
        raise FileNotFoundError(f"No vector store found at path: {path}")

    try:
        vector_store = FAISS.load_local(
            folder_path=path,
            embeddings=emb,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded from {path} ({vector_store.index.ntotal} items).")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store from {path}: {e}")
        raise e

def get_or_create_vector_store(
    embedding_model=None,
    force_rebuild: bool = False,
    path: str = config.VECTOR_DB_PATH
) -> FAISS:
    """
    Retrieves existing vector store from disk, or ingests, chunks, and builds a new one.
    """
    emb = embedding_model or get_embedding_model()

    if vector_store_exists(path) and not force_rebuild:
        return load_vector_store(embedding_model=emb, path=path)

    logger.info("Initializing vector store from policy documents...")
    docs = load_hr_policy_documents()
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks, emb)
    save_vector_store(vector_store, path)
    return vector_store

def get_retriever_from_faiss(
    vector_store: Optional[FAISS] = None,
    path: str = config.VECTOR_DB_PATH,
    top_k: int = config.TOP_K_RESULT
):
    """
    Creates and returns a retriever from FAISS vector store.
    """
    vs = vector_store or get_or_create_vector_store(path=path)
    return vs.as_retriever(search_kwargs={"k": top_k})