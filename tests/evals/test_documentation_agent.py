from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import assert_test
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
import pytest
import os

import src.graph as graph

current_dir = os.path.dirname(os.path.abspath(__file__))
test_documentation_agent = os.path.join(current_dir, "../datasets/documentation_intents.json")

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(test_documentation_agent)

# Test the agent orchestrator with the provided dataset
@pytest.mark.parametrize("golden", dataset.goldens)
def test_documentation_agent(golden: Golden):
    """
    Test the documentation agent through the graph and assert the expected output.
    """
    result = graph.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": "test_thread"},
            "callbacks": [CallbackHandler()]},
    )
    print("Test documentation agent result against golden expected output\n")
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])

    print("Test backend agent result\n")
    assert result["last_agent"] == "router_agent"
    assert result["next_agent"] == "documentation_agent"
    