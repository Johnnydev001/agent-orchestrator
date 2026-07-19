from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver

from .agents import run_router_agent, run_frontend_deep_agent
from .utils import route_from_supervisor
from .state import MessagesState

# Used for checkpointing and supervision tool calls with human in the loop
memory_saver = InMemorySaver()

builder = StateGraph(MessagesState)

builder.add_node("frontend_developer_agent", run_frontend_deep_agent)
builder.add_node("router_agent", run_router_agent)

builder.add_edge(START, "router_agent")
builder.add_conditional_edges(
	"router_agent",
	route_from_supervisor,
	{"frontend_developer_agent": "frontend_developer_agent", END: END},
)

_graph = builder.compile(checkpointer=memory_saver)

# Backwards-compatible code for invoking the graph and getting its state (for testing)
def invoke(*args, **kwargs):
	return _graph.invoke(*args, **kwargs)

def get_state(*args, **kwargs):
	return _graph.get_state(*args, **kwargs)

graph = _graph
