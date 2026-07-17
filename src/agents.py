import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from e2b import Sandbox
from langchain_e2b import E2BSandbox
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import MessagesState, RouterState

load_dotenv()

# Initialize E2B sandbox environment
e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)

# Initialize the model with the specified parameters
open_ai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o", openai_api_key=open_ai_api_key)

print("Model initialized successfully!\n")

frontend_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior Frontend Developer. Capable of creating optimized, responsive and clean UIs using tools such as React and Vue. Always use TypeScript for frontend code. For styling use CSS. Create the necessary files inside /home/user/output/frontend in the E2B sandbox, not /frontend. When you have code to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    },
    #tools=[fail_on_purpose] Leave this commented out for now, since it was used for testing errors
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Tech Lead. Based on the conversation, decide who should act next. "
            "If the task requires UI, route to 'frontend_developer_agent'. If the work is "
            "done and the app is complete, route to 'FINISH'. If an agent just responded "
            "but hasn't completed their work yet, you can route back to them. The last "
            "agent was: {last_agent}.",
        ),
        MessagesPlaceholder("messages"),
    ]
)
router_chain = router_prompt | model.with_structured_output(RouterState)

def run_frontend_deep_agent(state: MessagesState):
    invocation_input = state
    response = frontend_agent.invoke(invocation_input)
    print(f"Frontend agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "frontend_developer_agent",
    }

def run_router_agent(state: MessagesState):
    invocation_input = {
        "messages": state["messages"],
        "last_agent": state.get("last_agent", "No agent has responded yet"),
    }
    response = router_chain.invoke(invocation_input)
    print(f"Router agent output: {response}\n")

    return {
        "next_agent": response.next_agent,
        "last_agent": "router"
    }
