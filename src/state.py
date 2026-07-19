from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Literal

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    last_agent: str
    next_agent: str

class RouterState(BaseModel):
    next_agent: Literal["frontend_developer_agent", "FINISH"]
