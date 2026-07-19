import pytest

# Creates a test fixture that compiles the graph and makes it available for tests
@pytest.fixture(scope="session")
def compile_graph():
    from src import graph
    return graph

