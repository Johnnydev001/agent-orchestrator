import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from e2b import Sandbox
from langchain_e2b import E2BSandbox
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .state import MessagesState, RouterState

load_dotenv()

# Initialize E2B sandbox environment
e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)

# Initialize the model with the specified parameters
open_ai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-5.4-nano", openai_api_key=open_ai_api_key)

print("Model initialized successfully!\n")


# Agent creations
frontend_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior Frontend Developer. Capable of creating optimized, responsive and clean UIs using tools such as React and Vue. Always use TypeScript for frontend code. For styling use CSS. Create the necessary files inside /home/user/output/frontend in the E2B sandbox, not /frontend. When you have code to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    },
)

backend_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior Backend Developer. Capable of creating optimized, secure and clean backend systems using tools such as Node.js, Python, and Java. Always use best practices for backend code. Create the necessary files inside /home/user/output/backend in the E2B sandbox, not /backend. When you have code to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    }
)

devops_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior DevOps Engineer. Capable of creating optimized, secure and clean CI/CD pipelines and infrastructure using tools such as Docker, Kubernetes, and Terraform. Always use best practices for DevOps code. Create the necessary files inside /home/user/output/devops in the E2B sandbox, not /devops. When you have code to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    }
)

ui_designer_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior UI Designer. Capable of creating optimized, responsive and clean UI designs using tools such as Figma, Sketch, and Adobe XD. Always use best practices for UI design. Create the necessary files inside /home/user/output/ui in the E2B sandbox, not /ui. When you have design files to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    }
)

documentation_agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a Senior Documentation Writer. Capable of creating optimized, clear and concise documentation using tools such as Markdown, AsciiDoc, and reStructuredText. Always use best practices for documentation. Create the necessary files inside /home/user/output/docs in the E2B sandbox, not /docs. When you have documentation files to write, use the write_file tool.",
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]}
    }
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Tech Lead. Based on the conversation, decide who should act next. "
            "If the task requires UI design, route to 'ui_designer_agent'."
            "If the task requires frontend development, route to 'frontend_agent'."
            "If the task requires backend development, route to 'backend_agent'."
            "If the task requires DevOps work, route to 'devops_agent'."
            "If the task requires documentation work, route to 'documentation_agent'."
            " If the work is done and the app is complete, route to 'FINISH'. If an agent just responded "
            "but hasn't completed their work yet, you can route back to them. The last "
            "agent was: {last_agent}.",
        ),
        MessagesPlaceholder("messages"),
    ]
)
router_chain = router_prompt | model.with_structured_output(RouterState)

# Agent execution functions
def run_frontend_agent(state: MessagesState):
    invocation_input = state
    response = frontend_agent.invoke(invocation_input)
    print(f"Frontend agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "frontend_agent",
    }

def run_backend_agent(state: MessagesState):
    invocation_input = state
    response = backend_agent.invoke(invocation_input)
    print(f"Backend agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "backend_agent",
    }

def run_devops_agent(state: MessagesState):
    invocation_input = state
    response = devops_agent.invoke(invocation_input)
    print(f"DevOps agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "devops_agent",
    }

def run_ui_designer_agent(state: MessagesState):
    invocation_input = state
    response = ui_designer_agent.invoke(invocation_input)
    print(f"UI Designer agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "ui_designer_agent",
    }

def run_documentation_agent(state: MessagesState):
    invocation_input = state
    response = documentation_agent.invoke(invocation_input)
    print(f"Documentation agent output: {response}\n")

    return {
        "messages": response["messages"],
        "last_agent": "documentation_agent",
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
        "last_agent": "router_agent"
    }
