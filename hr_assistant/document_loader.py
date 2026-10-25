import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from hr_assistant import config

logger = logging.getLogger(__name__)

def load_hr_policy_documents(file_path: Optional[str] = None) -> List[Document]:
    """
    Loads HR policy documents from the specified file path.
    
    Args:
        file_path: Path to the HR policy document. Defaults to config.HR_POLICY_DATAFILE.
        
    Returns:
        List of loaded LangChain Document objects.
    """
    target_path = file_path or config.HR_POLICY_DATAFILE
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Document file does not exist at: {target_path}")

    try:
        text_loader = TextLoader(target_path, encoding="utf-8")
        documents = text_loader.load()
    except UnicodeDecodeError:
        logger.warning("Failed to decode with utf-8, attempting fallback to latin-1.")
        text_loader = TextLoader(target_path, encoding="latin-1")
        documents = text_loader.load()
    except Exception as e:
        logger.error(f"Error loading documents from {target_path}: {e}")
        raise e

    if not documents:
        raise ValueError(f"No document content could be extracted from {target_path}")

    return documents