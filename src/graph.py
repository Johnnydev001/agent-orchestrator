from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agents import run_router_agent, run_frontend_deep_agent
from tools import write_to_file
from utils import route_based_on_last_message, route_from_supervisor
from state import MessagesState

# Custom tool node to execute the write_to_file tool and save results in state
tools_node = ToolNode([write_to_file])

# Used for checkpointing and supervision tool calls with human in the loop
memory_saver = MemorySaver()

builder = StateGraph(MessagesState)

builder.add_node("frontend_developer_agent", run_frontend_deep_agent)
builder.add_node("router_agent", run_router_agent)
builder.add_node("tools", tools_node)

builder.add_edge(START, "router_agent")
builder.add_edge("tools", "router_agent")
builder.add_conditional_edges(
	"router_agent",
	route_from_supervisor,
	{"frontend_developer_agent": "frontend_developer_agent", END: END},
)
builder.add_conditional_edges(
	"frontend_developer_agent",
	route_based_on_last_message,
	{"tools": "tools", "router": "router_agent"},
)

graph = builder.compile(checkpointer=memory_saver,  interrupt_before=["tools"])
