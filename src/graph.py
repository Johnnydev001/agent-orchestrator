from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agents import frontend_developer_agent, backend_developer_agent, router
from tools import write_to_file
from state import MessagesState
from utils import route_based_on_last_message, route_from_supervisor

# Custom tool node to execute the write_to_file tool and save results in state
tools_node = ToolNode([write_to_file])

# Used for checkpointing and supervision tool calls with human in the loop
memory_saver = MemorySaver()

builder = StateGraph(MessagesState)

builder.add_node("router", router)
builder.add_node("frontend_developer_agent", frontend_developer_agent)
builder.add_node("backend_developer_agent", backend_developer_agent)
builder.add_node("tools", tools_node)

# Add conditional edges from agents - if they call tools, go to tools node, else go to router
builder.add_conditional_edges("frontend_developer_agent", route_based_on_last_message, {"tools": "tools", "router": "router"})
builder.add_conditional_edges("backend_developer_agent", route_based_on_last_message, {"tools": "tools", "router": "router"})

builder.add_edge("tools", "router") # After tools execute, go back to router
builder.add_conditional_edges(
	"router",
	route_from_supervisor,
	{
		"frontend_developer_agent": "frontend_developer_agent",
		"backend_developer_agent": "backend_developer_agent",
		END: END,
	},
)
builder.add_edge(START, "router")

graph = builder.compile(checkpointer=memory_saver,  interrupt_before=["tools"])
