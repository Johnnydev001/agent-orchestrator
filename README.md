# Agent Orchestrator

## Project Description

Agent Orchestrator is a self-learning and experimentation repository for building and coordinating AI agents. It explores how LangGraph, LangChain, and an OpenAI model can be combined with an E2B sandbox to route tasks and generate working output.

## Work in Progress

This project is a work in progress. More agents and functionality will be added in the future.

## Repository Structure

- `src/` - Python source code for the orchestrator, graph logic, agent setup, state, tools, and utilities.

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- OpenAI
- E2B sandbox
- python-dotenv

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key
- E2B API key

```bash
git clone https://github.com/Johnnydev001/agent-orchestrator.git
cd agent-orchestrator
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` and `E2B_API_KEY` in `.env` to your OpenAI and E2B API keys.

## How to Run

```bash
uv run python src/main.py "create a simple HTML page"
```

Write operations may pause and ask for approval in the terminal before continuing.
