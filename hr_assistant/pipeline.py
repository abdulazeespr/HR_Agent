import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Optional, List, Dict, Any

from hr_assistant import config
from hr_assistant.document_loader import load_hr_policy_documents
from hr_assistant.splitter import split_documents
from hr_assistant.embeddings import get_embedding_model
from hr_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_or_create_vector_store,
    get_retriever_from_faiss
)
from hr_assistant.tools import create_policy_search_tool
from hr_assistant.llm import get_llm
from hr_assistant.agent import create_hr_agent, invoke_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hr_assistant.pipeline")

def build_vector_store_documents(
    file_path: Optional[str] = None,
    force_rebuild: bool = False,
    jina_api_key: Optional[str] = None
):
    """
    Ingests HR policy documents, splits them into semantic chunks,
    generates embeddings, and saves a FAISS vector index to disk.
    """
    try:
        path = config.VECTOR_DB_PATH
        if vector_store_exists(path) and not force_rebuild:
            logger.info(f"Vector store already exists at {path}. Use force_rebuild=True to overwrite.")
            emb = get_embedding_model(api_key=jina_api_key)
            return load_vector_store(embedding_model=emb, path=path)

        logger.info(f"Loading documents from {file_path or config.HR_POLICY_DATAFILE}...")
        documents = load_hr_policy_documents(file_path=file_path)
        
        logger.info("Splitting documents into chunks...")
        chunks = split_documents(documents)

        logger.info("Initializing embedding model...")
        embedding_model = get_embedding_model(api_key=jina_api_key)

        logger.info("Building FAISS vector index...")
        vector_store = build_vector_store(chunks, embedding_model)

        logger.info("Saving vector index to disk...")
        save_vector_store(vector_store, path)
        return vector_store

    except Exception as e:
        logger.error(f"Failed to build vector store documents: {e}")
        raise e

class HRAssistantPipeline:
    """
    End-to-end pipeline orchestrator for Acme Corp HR Policy Assistant.
    """
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        force_rebuild_index: bool = False
    ):
        self.jina_api_key = jina_api_key or config.JINA_API_KEY
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        self.llm_model = llm_model or config.LLM_MODEL
        self.temperature = temperature

        # 1. Initialize Vector Store & Retriever
        self.embedding_model = get_embedding_model(api_key=self.jina_api_key)
        self.vector_store = get_or_create_vector_store(
            embedding_model=self.embedding_model,
            force_rebuild=force_rebuild_index
        )
        self.retriever = get_retriever_from_faiss(self.vector_store)

        # 2. Initialize Tool & LLM
        self.search_tool = create_policy_search_tool(self.retriever)
        self.llm = get_llm(
            api_key=self.groq_api_key,
            model=self.llm_model,
            temperature=self.temperature
        )

        # 3. Create Agent
        self.agent = create_hr_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=config.SYSTEM_PROMPT
        )
        logger.info("HRAssistantPipeline initialized successfully.")

    def ask(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Executes query through the agent and returns the answer with citations.
        """
        return invoke_agent(self.agent, question, chat_history=chat_history)

    def search_policies(self, query: str, k: int = config.TOP_K_RESULT) -> List[Any]:
        """
        Performs direct similarity search without agent reasoning.
        """
        return self.vector_store.similarity_search(query, k=k)

    def rebuild_index(self, file_path: Optional[str] = None):
        """
        Re-indexes the policy document from scratch.
        """
        self.vector_store = build_vector_store_documents(
            file_path=file_path,
            force_rebuild=True,
            jina_api_key=self.jina_api_key
        )
        self.retriever = get_retriever_from_faiss(self.vector_store)
        self.search_tool = create_policy_search_tool(self.retriever)
        self.agent = create_hr_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=config.SYSTEM_PROMPT
        )
        return True


def run_pipeline_test():
    """CLI test routine for validating indexing and search."""
    print("=== Testing HR Assistant Vector Indexing ===")
    vs = build_vector_store_documents(force_rebuild=True)
    print(f"Vector store successfully built! Total vectors: {vs.index.ntotal}")

    test_queries = [
        "What is the annual leave policy?",
        "How many days can I work from home?",
        "What is the notice period during probation?"
    ]
    print("\n=== Testing Similarity Search ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = vs.similarity_search(q, k=2)
        for i, doc in enumerate(results, start=1):
            print(f"  Match {i}: {doc.page_content[:120]}...")

    print("\n[OK] Pipeline vector indexing and retrieval verified successfully.")

if __name__ == "__main__":
    run_pipeline_test()