import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
QROQ_API_KEY = GROQ_API_KEY  # Backward compatibility alias
JINA_API_KEY = os.getenv('JINA_API_KEY', '')

# Model Configurations
DEFAULT_LLM_MODEL = 'openai/gpt-oss-120b'
LLM_MODEL = os.getenv('LLM_MODEL', DEFAULT_LLM_MODEL)
AVAILABLE_MODELS = [
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen3.8-27b'
]
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'jina-embeddings-v2-base-en')

# Data & Storage Paths
HR_POLICY_DATAFILE = os.path.join(BASE_DIR, 'data', 'hr_policy.txt')
HR_POCILY_DATAFILE = HR_POLICY_DATAFILE  # Backward compatibility alias
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'data', 'vector_db')

# Chunking & Retrieval Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULT = 3

# System Prompt
SYSTEM_PROMPT = """You are a helpful and professional HR Assistant AI for Acme Corp.
Your goal is to accurately answer employee questions based ONLY on the company HR policy context provided.

Guidelines:
1. Carefully review the retrieved context before answering.
2. Formulate clear, concise, and polite answers citing relevant policy section names where applicable.
3. If the answer cannot be determined from the provided context, state:
   "I could not find the answer in the provided HR policy handbook. Please consult the HR department for further assistance."
4. Do not speculate or use outside knowledge beyond the provided policy documents.
"""

def check_env_variable(require_groq: bool = True, require_jina: bool = True) -> dict:
    """
    Validates environment configuration.
    Returns a status dictionary with availability details.
    """
    status = {
        "groq_valid": bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("jina_")),
        "jina_valid": bool(JINA_API_KEY),
        "data_file_exists": os.path.exists(HR_POLICY_DATAFILE),
        "vector_db_exists": os.path.exists(VECTOR_DB_PATH),
    }

    if require_jina and not status["jina_valid"]:
        raise ValueError("JINA_API_KEY not found or invalid in environment.")
    if require_groq and not status["groq_valid"]:
        raise ValueError("GROQ_API_KEY not found or invalid in environment.")
    if not status["data_file_exists"]:
        raise FileNotFoundError(f"HR policy file not found at {HR_POLICY_DATAFILE}")

    return status