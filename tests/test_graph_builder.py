import pytest
import os
import networkx as nx
from src.graph_builder import GraphBuilder
from src.utils import load_json

@pytest.fixture
def sample_history():
    # Load or mock history data
    sample_path = 'tests/sample_history.json'  # Assume a sample JSON exists or create mock
    if os.path.exists(sample_path):
        return load_json(sample_path)
    else:
        return {
            'metadata': {'name': 'test_repo'},
            'commits': [
                {'sha': 'abc123', 'author': 'user1', 'date': '2023-01-01', 'message': 'Initial commit',
                 'files': [{'filename': 'file1.py', 'status': 'added', 'additions': 10, 'deletions': 0, 'changes': 10, 'diff': ''}]}
            ]
        }

@pytest.fixture
def builder(sample_history):
    return GraphBuilder(sample_history, cache_dir='tests/cache')

def test_build_graph(builder):
    G = builder.get_networkx_graph()
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    assert 'user1' in G.nodes
    assert G.nodes['user1']['type'] == 'author'
    assert 'abc123' in G.nodes
    assert 'file1.py' in G.nodes

def test_to_pytorch_geometric(builder):
    pyg_data = builder.to_pytorch_geometric()
    assert pyg_data.num_nodes > 0
    assert pyg_data.edge_index.size(1) > 0
    assert pyg_data.x.size(0) == pyg_data.num_nodes

def test_cache(builder):
    G = builder.get_networkx_graph()
    cached_G = nx.read_gpickle(builder.graph_cache_path)
    assert nx.is_isomorphic(G, cached_G)

# Add more tests: edge cases, large data, etc.4