from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Literal

class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_agent: str
    next_agent: str

class RouterState(BaseModel):
    next_agent: Literal["frontend_developer_agent", "backend_developer_agent", "FINISH"]
