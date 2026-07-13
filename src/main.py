import argparse
import json
import sys
from collections.abc import Mapping

from langchain.messages import HumanMessage

from graph import graph


def _print_stream_output(output):
    if not isinstance(output, Mapping):
        return

    for node_name, state_update in output.items():
        if not isinstance(state_update, Mapping):
            continue

        messages = state_update.get("messages", [])
        if messages:
            print(f"Messages from {node_name}: {messages[0].content}\n")


def _pending_tool_calls(snapshot):
    if snapshot.next != ("tools",):
        raise RuntimeError(f"Unexpected paused graph state: {snapshot.next!r}")

    messages = snapshot.values.get("messages")
    if not messages:
        raise RuntimeError("Paused graph state has no messages")

    tool_calls = getattr(messages[-1], "tool_calls", None)
    if not tool_calls:
        raise RuntimeError("Paused graph state has no pending tool calls")

    pending_calls = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, Mapping):
            raise RuntimeError(f"Tool call {index} is malformed")

        name = tool_call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"Tool call {index} has no valid name")

        arguments = tool_call.get("args")
        try:
            serialized_arguments = json.dumps(arguments, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Arguments for tool call {index} ({name}) are not JSON-serializable"
            ) from error

        pending_calls.append((name, serialized_arguments))

    return pending_calls


def _approve_batch(pending_calls):
    print("Pending tool calls:")
    for index, (name, serialized_arguments) in enumerate(pending_calls, start=1):
        print(f"\nTool call {index}: {name}")
        print("Arguments:")
        print(serialized_arguments)

    for index, (name, _) in enumerate(pending_calls, start=1):
        try:
            answer = input(
                f"\nApprove tool call {index}/{len(pending_calls)} ({name})? [y/yes]: "
            )
        except EOFError:
            print("\nApproval rejected: input ended before explicit approval.", file=sys.stderr)
            return False

        if answer.strip().lower() not in {"y", "yes"}:
            print(
                f"Approval rejected for tool call {index} ({name}).",
                file=sys.stderr,
            )
            return False

    return True


def run(prompt, orchestrator_graph=graph):
    print(f"Running the agent orchestrator with the following prompt: {prompt}")

    inputs = {"messages": [HumanMessage(content=prompt)]}
    config = {"configurable": {"thread_id": "thread_1"}}

    print("Running graph with prompt:", prompt + "\n")

    graph_input = inputs
    while True:
        try:
            for output in orchestrator_graph.stream(graph_input, config):
                _print_stream_output(output)
            snapshot = orchestrator_graph.get_state(config)
        except Exception as error:
            phase = "initial graph run" if graph_input is inputs else "graph resume"
            print(f"Failed during {phase}: {error}", file=sys.stderr)
            return 1

        if not snapshot.next:
            return 0

        try:
            pending_calls = _pending_tool_calls(snapshot)
        except RuntimeError as error:
            print(f"Malformed interrupted graph state: {error}", file=sys.stderr)
            return 1

        if not _approve_batch(pending_calls):
            return 2

        graph_input = None


def main():
    arg_parser = argparse.ArgumentParser(description="Run the agent orchestrator.")
    arg_parser.add_argument("prompt", type=str, help="The prompt to send to the agent orchestrator.")
    args = arg_parser.parse_args()

    try:
        return run(args.prompt)
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
