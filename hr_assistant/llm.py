import logging
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from hr_assistant import config

logger = logging.getLogger(__name__)

def get_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1
) -> ChatGroq:
    """
    Initializes and returns the ChatGroq model instance.
    
    Args:
        api_key: Optional Groq API Key override.
        model: Model name string. Defaults to config.LLM_MODEL.
        temperature: Model temperature. Defaults to 0.1 for high factual accuracy.
        
    Returns:
        Configured ChatGroq instance.
    """
    target_key = api_key or config.GROQ_API_KEY
    target_model = model or config.LLM_MODEL

    if not target_key or target_key.startswith("jina_"):
        raise ValueError(
            "A valid Groq API key is required. Please set GROQ_API_KEY in your .env or configure it in the UI."
        )

    try:
        llm = ChatGroq(
            api_key=target_key,
            model=target_model,
            temperature=temperature
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        raise e

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Returns the standard ChatPromptTemplate for HR policy QA.
    """
    return ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nRelevant Policy Context:\n{context}\n\nPlease provide a clear and concise answer based strictly on the above context.")
    ])

# Backward compatibility alias
get_prompt = get_rag_prompt