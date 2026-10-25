import logging
from typing import List, Optional
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def create_policy_search_tool(retriever) -> BaseTool:
    """
    Creates an HR policy search tool bound to a retriever.
    """
    @tool("search_hr_policy")
    def search_hr_policy(query: str) -> str:
        """Searches the official Acme Corp HR Policy Handbook for rules, guidelines, leave policies, work-from-home rules, notice periods, and benefits."""
        try:
            docs: List[Document] = retriever.invoke(query)
            if not docs:
                return "No matching HR policy information found."

            formatted_snippets = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "HR Policy Handbook")
                content = doc.page_content.strip()
                formatted_snippets.append(f"[Excerpt {i} | Source: {source}]\n{content}")

            return "\n\n---\n\n".join(formatted_snippets)
        except Exception as e:
            logger.error(f"Error executing HR policy search tool: {e}")
            return f"Error querying policy documents: {str(e)}"

    return search_hr_policy

# Backward compatibility alias
create_sreach_tool = create_policy_search_tool