from langgraph.graph import StateGraph
from langgraph.graph import START, END

from agents import frontend_developer_agent, backend_developer_agent, router
from state import MessagesState

# Initialize builder with the MessagesState 
builder = StateGraph(MessagesState)

builder.add_node("backend_developer_agent", backend_developer_agent)
builder.add_node("frontend_developer_agent", frontend_developer_agent)
builder.add_node("router", router)

builder.add_edge(START, "frontend_developer_agent")
builder.add_edge("frontend_developer_agent", END)

graph = builder.compile()
