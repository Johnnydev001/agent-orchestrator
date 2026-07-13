import argparse
from langchain.messages import HumanMessage
from graph import graph

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run the agent orchestrator.")
    arg_parser.add_argument("prompt", type=str, help="The prompt to send to the agent orchestrator.")
    args = arg_parser.parse_args()

    print(f"Running the agent orchestrator with the following prompt: {args.prompt}")

    inputs = {
        "messages": [HumanMessage(content=args.prompt)]
    }

    config = {"configurable": {"thread_id": "thread_1"}}

    print("Running graph with prompt:", args.prompt + "\n")

    for output in graph.stream(inputs, config):
        print(f"Output: {output}")
