from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_agent: str
    next_agent: str