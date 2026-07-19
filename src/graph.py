from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver

from .agents import run_router_agent, run_frontend_agent, run_backend_agent, run_ui_designer_agent, run_documentation_agent, run_devops_agent
from .utils import route_from_supervisor
from .state import MessagesState

# Used for checkpointing and supervision tool calls with human in the loop
memory_saver = InMemorySaver()
builder = StateGraph(MessagesState)

agents_names: list[str] = ["router_agent", "frontend_agent", "backend_agent", "devops_agent", "ui_designer_agent", "documentation_agent"]

for agent_name in agents_names:
	builder.add_node(agent_name, eval("run_" + agent_name))

builder.add_edge(START, "router_agent")
builder.add_conditional_edges(
	"router_agent",
	route_from_supervisor,
	{
		"frontend_agent": "frontend_agent",
		"backend_agent": "backend_agent",
		"devops_agent": "devops_agent",
		"ui_designer_agent": "ui_designer_agent",
		"documentation_agent": "documentation_agent",
		END: END,
	},
)

_graph = builder.compile(checkpointer=memory_saver)

# Backwards-compatible code for invoking the graph and getting its state (for testing)
def invoke(*args, **kwargs):
	return _graph.invoke(*args, **kwargs)

def get_state(*args, **kwargs):
	return _graph.get_state(*args, **kwargs)

graph = _graph
