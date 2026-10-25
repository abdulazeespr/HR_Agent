import logging
from typing import Optional, List, Dict, Any
from langchain.agents import create_agent as lc_create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from hr_assistant import config
from hr_assistant import llm as llm_module
from hr_assistant import tools as tools_module
from hr_assistant import vector_store as vs_module

logger = logging.getLogger(__name__)

def create_hr_agent(
    model: Optional[BaseChatModel] = None,
    tools: Optional[List[Any]] = None,
    system_prompt: Optional[str] = None
):
    """
    Creates a modern LangChain agent graph equipped with HR policy search tools.
    
    Args:
        model: Optional pre-configured chat model. If None, initializes via get_llm().
        tools: Optional list of tools. If None, creates policy search tool.
        system_prompt: System prompt instructing the agent. Defaults to config.SYSTEM_PROMPT.
        
    Returns:
        CompiledStateGraph agent instance.
    """
    try:
        chat_model = model or llm_module.get_llm()
        
        if tools is None:
            retriever = vs_module.get_retriever_from_faiss()
            search_tool = tools_module.create_policy_search_tool(retriever)
            active_tools = [search_tool]
        else:
            active_tools = tools

        prompt_str = system_prompt or config.SYSTEM_PROMPT

        # Modern LangChain 1.4 create_agent graph
        agent_graph = lc_create_agent(
            model=chat_model,
            tools=active_tools,
            system_prompt=prompt_str
        )
        return agent_graph
    except Exception as e:
        logger.error(f"Error creating HR agent: {e}")
        raise e

def invoke_agent(agent_graph, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Invokes the agent graph with conversation history and parses output.
    
    Returns:
        Dict containing:
          - 'answer': string response
          - 'tool_calls': list of tool invocations made during execution
          - 'sources': list of excerpts retrieved
    """
    messages = []
    
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_query))

    result = agent_graph.invoke({"messages": messages})
    all_messages = result.get("messages", [])

    final_answer = ""
    tool_calls_info = []
    sources = []

    for msg in all_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                final_answer = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_info.append({
                        "name": tc.get("name"),
                        "args": tc.get("args")
                    })
        elif isinstance(msg, ToolMessage):
            sources.append(msg.content)

    return {
        "answer": final_answer or "I could not generate an answer.",
        "tool_calls": tool_calls_info,
        "sources": sources,
        "raw_messages": all_messages
    }

# Backward compatibility alias
create_agent = create_hr_agent