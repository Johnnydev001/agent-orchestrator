from langgraph.graph import StateGraph
from langgraph.graph import START

from agents import frontend_developer_agent, backend_developer_agent, router
from state import MessagesState
from utils import route_from_supervisor

# Initialize builder with the MessagesState 
builder = StateGraph(MessagesState)

builder.add_node("frontend_developer_agent", frontend_developer_agent)
builder.add_node("backend_developer_agent", backend_developer_agent)
builder.add_node("router", router)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_from_supervisor)

graph = builder.compile()
