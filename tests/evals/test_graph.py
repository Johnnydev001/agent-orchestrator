from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import assert_test
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
import pytest
import os

import src.graph as graph

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "../datasets/frontend_intents.json")

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(dataset_path)

# Test the agent orchestrator with the provided dataset
@pytest.mark.parametrize("golden", dataset.goldens)
def test_frontend_agent(golden: Golden):
    """
    Test the frontend agent through the graph and assert the expected output.
    """
    result = graph.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": "test_thread"},
            "callbacks": [CallbackHandler()]},
    )
    assert_test(golden=golden, metrics=[TaskCompletionMetric()])
    assert result["last_agent"] == "router"
    assert result["next_agent"] == "frontend_developer_agent"
    