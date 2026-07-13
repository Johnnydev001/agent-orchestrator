import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import MessagesState


load_dotenv()

# Initialize the model with the specified parameters
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=1000, openai_api_key=api_key)

print("Model initialized successfully")


def _run_agent(state: MessagesState, system_prompt: str, last_agent: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = llm.invoke(prompt.invoke(state))
    return {
        "messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))],
        "last_agent": last_agent,
    }

# Agents definitions
def frontend_developer_agent(state: MessagesState) -> dict:
    """
    A frontend developer agent specialized in creating user interfaces and experiences. It can provide guidance on frontend technologies, frameworks, and best practices.
    """
    return _run_agent(
        state,
        "You are a frontend developer agent: expert in building beautiful, accessible, and high-performance web applications using React, Next.js, and TypeScript. Produce clean, maintainable, component-based code with strong TypeScript types and modern CSS patterns (CSS Modules, Tailwind, or CSS-in-JS). Prioritize WCAG accessibility, semantic HTML, ARIA attributes, and keyboard navigation. Optimize performance via code-splitting, SSR/SSG choices, caching, and minimal bundle sizes. Provide responsive, pixel-perfect UIs and Figma-ready design mockups (component list, tokens, spacing, colors, and exportable assets). When returning solutions include complete file examples, folder structure, TypeScript types, brief rationale, testing suggestions, and step-by-step integration instructions.",
        "frontend_developer_agent",
    )

def backend_developer_agent(state: MessagesState) -> dict:
    """
    A backend developer agent specialized in server-side logic, databases, and APIs. It can provide guidance on backend technologies, frameworks, and best practices.
    """
    return _run_agent(
        state,
        "You are a backend developer agent. You specialize in server-side logic, databases, and APIs.",
        "backend_developer_agent",
    )

def router(state: MessagesState) -> dict:
    """
    A router agent that decides which agent should handle the next step based on the current state and messages. It can route tasks to the appropriate agent based on their expertise.
    """
    last_agent = state.get("last_agent", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a router agent. You decide which agent should handle the next step based on the current state and messages. "
        f"The last agent that executed was {last_agent}. You should route the next task to the appropriate agent based on their expertise. "
        f"if the last agent was 'backend_developer_agent', route to 'frontend_developer_agent'. "
        f"if the last agent was 'START', route to 'backend_developer_agent'. "
        f"if the last agent was 'frontend_developer_agent', route to 'backend_developer_agent'."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = llm.invoke(prompt.invoke(state))
    return {
        "messages": [response if isinstance(response, AIMessage) else AIMessage(content=str(response))],
        "next_agent": response.content if isinstance(response, AIMessage) else str(response),
    }

