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

    # Each output object represents the state updates from each node in the graph for a single step of execution. 
    # The graph will continue to execute until it reaches a terminal state (END).
    for output in graph.stream(inputs, config):
        for node_name, state_update in output.items():
            messages = state_update.get("messages", [])[0].content

            if(messages):
                print(f"Messages from {node_name}: {messages}\n")
