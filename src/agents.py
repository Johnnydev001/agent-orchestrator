import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import MessagesState, RouterState
from tools import write_to_file

load_dotenv()

# Initialize the model with the specified parameters
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=1000, openai_api_key=api_key)

print("Model initialized successfully")

# Agents definitions
def backend_developer_agent(state: MessagesState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Backend Developer. Write efficient, secure Python code. "
                   "Only focus on the server-side logic. Do not write frontend code. Create the necessary files in output/backend. When you have code to write, use the write_to_file tool. Do not respond with explanations without making tool calls."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    llm_with_tools = llm.bind_tools([write_to_file])
    chain = prompt | llm_with_tools
    response = chain.invoke(state)
    return {
        "messages": [response],
        "last_agent": "backend_developer_agent"
    }

def frontend_developer_agent(state: MessagesState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Frontend Developer. Capable of creating optimized, responsive and clean UIs using tools such as React and Vue. Always use TypeScript for frontend code. For styling use Tailwind CSS. Only focus on the UI/UX. Do not write server code. Create the necessary files in output/frontend. When you have code to write, use the write_to_file tool. Do not respond with explanations without making tool calls."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    llm_with_tools = llm.bind_tools([write_to_file])
    chain = prompt | llm_with_tools
    response = chain.invoke(state)
    return {
        "messages": [response],
        "last_agent": "frontend_developer_agent"
    }

def router(state: MessagesState) -> dict:
    last_agent = state.get("last_agent", "none")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a Tech Lead. Based on the conversation, decide who should act next. "
                   f"Last agent that executed: {last_agent}. "
                   "If the task requires UI, route to 'frontend_developer_agent'. "
                   "If it requires data/APIs, route to 'backend_developer_agent'. "
                   "If both are done and the app is complete, route to 'FINISH'. "
                   "If an agent just responded but hasn't completed their work yet, you can route back to them."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    router_chain = prompt | llm.with_structured_output(RouterState)
    response = router_chain.invoke(state)
    return {
        "next_agent": response.next_agent
    }
