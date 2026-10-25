import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Splits documents into manageable chunks for vector search.
    
    Args:
        documents: List of LangChain Documents.
        chunk_size: Size of each text chunk. Defaults to config.CHUNK_SIZE.
        chunk_overlap: Overlap between consecutive chunks. Defaults to config.CHUNK_OVERLAP.
        
    Returns:
        List of chunked Document objects.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list.")

    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise e
