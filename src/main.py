import argparse
import sys
from uuid import uuid4

from graph import graph

def main():
    arg_parser = argparse.ArgumentParser(description="Run the agent orchestrator.")
    arg_parser.add_argument("prompt", type=str, help="The prompt to send to the agent orchestrator.")
    args = arg_parser.parse_args()

    print(f"Running agent orchestrator with prompt: {args.prompt}\n")

    try:
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": args.prompt}
            ]
        }, config=config)
        print(f"Graph invoke result: {result}\n")

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
