from langgraph.graph import END

from state import MessagesState

def route_based_on_last_message(state: MessagesState) -> str:
    """Check if the last message has tool calls."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "router"

def route_from_supervisor(state: MessagesState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]