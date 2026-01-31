# backend.py
"""
Robust backend for Streamlit + LangGraph demo.
- Uses environment variable OPENROUTER_API_KEY for LLM
- Uses MemorySaver (in-memory) for checkpointer (safer on Streamlit Cloud)
- Tries to enable DuckDuckGo search if the dependency is available,
  otherwise provides a safe fallback tool.
- Avoids crashing on missing optional packages.
"""

import os
import requests
from datetime import datetime
from typing import TypedDict, Annotated, List

# Optional dotenv (works locally only)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional (Streamlit secrets will be used in production)
    pass

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# -------------------
# 0. Checkpoint / Persistence
# -------------------
# Use MemorySaver for Streamlit-friendly in-memory persistence.
# (If you want persistent SQLite locally, do that only for self-hosted servers.)
try:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
except Exception:
    # In case the exact module path differs in some releases, try a fallback import
    try:
        # older/newer path attempt
        from langgraph.checkpoint import MemorySaver

        checkpointer = MemorySaver()
    except Exception:
        # Last resort: simple in-memory shim
        class _SimpleMemorySaver:
            def __init__(self):
                self._store = []

            def save(self, *args, **kwargs):
                self._store.append((args, kwargs))

            def list(self, *args, **kwargs):
                # minimal interface used by retrieve_all_threads below
                return []

        checkpointer = _SimpleMemorySaver()

# -------------------
# 1. LLM (OpenRouter / ChatOpenAI)
# -------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    # Fail fast with a clear message — Streamlit secrets should set this in production.
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please add it to Streamlit secrets or your environment.")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# -------------------
# 2. Tools
# -------------------

# 2.A Try to enable DuckDuckGoSearchRun (optional)
search_tool = None
try:
    from langchain_community.tools import DuckDuckGoSearchRun

    try:
        # Some environments require extra optional dependency `duckduckgo_search`.
        search_tool = DuckDuckGoSearchRun(region="us-en")
    except Exception as e:
        # If instantiation fails, keep search_tool = None and continue
        search_tool = None
except Exception:
    # dependency not available -> fallback will be used
    search_tool = None


# 2.B Define a fallback search tool (if DuckDuckGo is not available)
@tool
def search_fallback(query: str) -> dict:
    """
    Fallback: DuckDuckGoSearchRun not available in the environment.
    Returns a friendly message instead of failing the import.
    """
    return {
        "error": "Search tool is not available in this environment. "
        "Install required package `langchain-community` + `duckduckgo_search` locally, "
        "or enable the DuckDuckGo search tool in your deployment."
    }


# 2.C Calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


# 2.D Current time tool
@tool
def get_current_time() -> dict:
    now = datetime.now()
    return {"time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d")}


# 2.E Stock price tool (example). Uses ALPHAVANTAGE_API_KEY if provided.
@tool
def get_stock_price(symbol: str) -> dict:
    alpha_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not alpha_key:
        return {"error": "ALPHAVANTAGE_API_KEY not set. Set it in environment if you want live stock prices."}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_key}"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# 2.F Build tools list (use available search tool or fallback)
tools: List = []
if search_tool is not None:
    tools.append(search_tool)
else:
    tools.append(search_fallback)

tools.extend([get_stock_price, calculator, get_current_time])

# Bind tools to llm if available (some LLM wrappers support it)
try:
    llm_with_tools = llm.bind_tools(tools)
except Exception:
    # If bind_tools isn't available on this LLM wrapper, fall back to plain llm
    llm_with_tools = llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state.get("messages", [])
    # If using llm_with_tools supports `.invoke` with messages, use it; otherwise call llm directly.
    if hasattr(llm_with_tools, "invoke"):
        response = llm_with_tools.invoke(messages)
    else:
        # Fallback: call plain LLM (wrap messages into the expected call)
        response = llm(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer (already configured above)
# -------------------
# 'checkpointer' variable initialized earlier (MemorySaver or fallback shim)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    try:
        # checkpointer.list(...) may return iterable of checkpoints with config path used earlier
        for checkpoint in checkpointer.list(None):
            # defensive access; some saver implementations may differ
            cfg = getattr(checkpoint, "config", None) or checkpoint.get("config", None)
            if cfg:
                thread_id = cfg.get("configurable", {}).get("thread_id")
                if thread_id:
                    all_threads.add(thread_id)
    except Exception:
        # if the checkpointer doesn't support listing (fallback), return empty
        pass
    return list(all_threads)
