import argparse
import sys
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command, RunnableConfig

if __package__:
    from .agents import e2b_sandbox
    from .graph import graph
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.agents import e2b_sandbox
    from src.graph import graph

def main():
    arg_parser = argparse.ArgumentParser(description="Run the agent orchestrator.")
    arg_parser.add_argument("prompt", type=str, help="The prompt to send to the agent orchestrator.")
    args = arg_parser.parse_args()

    print(f"Running agent orchestrator with prompt: {args.prompt}\n")

    try:
        thread_id = str(uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": args.prompt}
            ]
        }, config=config)

        state = graph.get_state(config)

        if state.next:
            print("Next invocation:", state.next)

            for task in state.tasks:
                if task.interrupts:

                    print("Execution is paused: it requires approval.")
                    approval: str = input("Type 'y' or 'yes' to continue: ").strip().lower()

                    if approval in ("y", "yes"):
                        result = graph.invoke(
                            Command(
                                resume={
                                    "decisions": [
                                        {
                                            "type": "approve"
                                        }
                                    ]
                                }
                            ),
                            config=config,
                            version="v2"
                        )

                        print("Graph execution resumed after approval.")
                    else:
                        print("Execution left paused: approval was not entered.")

        graph_invoke_result = result
        print(f"Graph invoke result: {graph_invoke_result}\n")

        e2b_sandbox.kill()

    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
