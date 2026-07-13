
from langgraph.graph import END
from state import MessagesState

def route_from_supervisor(state: MessagesState) -> str:
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]