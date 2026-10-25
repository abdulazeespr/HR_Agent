import logging
from typing import Optional
from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

logger = logging.getLogger(__name__)

def get_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> JinaEmbeddings:
    """
    Initializes and returns the Jina Embeddings model.
    
    Args:
        model_name: Model identifier (defaults to config.EMBEDDING_MODEL_NAME)
        api_key: Jina API key (defaults to config.JINA_API_KEY)
        
    Returns:
        Configured JinaEmbeddings instance.
    """
    target_model = model_name or config.EMBEDDING_MODEL_NAME
    target_key = api_key or config.JINA_API_KEY

    if not target_key:
        raise ValueError(
            "Jina API Key is missing. Please provide it in your .env or pass it directly."
        )

    try:
        embedding_model = JinaEmbeddings(
            model_name=target_model,
            jina_api_key=target_key
        )
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize JinaEmbeddings: {e}")
        raise e