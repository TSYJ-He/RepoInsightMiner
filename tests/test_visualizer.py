import pytest
import os
from src.visualizer import Visualizer
from src.graph_builder import GraphBuilder
from src.miner import Miner
from src.utils import load_json

@pytest.fixture
def sample_history():
    # Mock or load
    return {
        'metadata': {'name': 'test_repo'},
        'commits': []
    }

@pytest.fixture
def builder(sample_history):
    return GraphBuilder(sample_history)

@pytest.fixture
def miner(builder):
    nx_g = builder.get_networkx_graph()
    pyg_d = builder.to_pytorch_geometric()
    return Miner(nx_g, pyg_d)

@pytest.fixture
def insights(miner, sample_history):
    return miner.mine_insights(sample_history)

@pytest.fixture
def visualizer(builder, insights):
    nx_g = builder.get_networkx_graph()
    return Visualizer(nx_g, insights, output_dir='tests/viz')

def test_generate_dashboard(visualizer):
    dashboard_path = 'tests/dashboard_test.py'
    visualizer.generate_dashboard(dashboard_path)
    assert os.path.exists(dashboard_path)
    with open(dashboard_path, 'r') as f:
        content = f.read()
    assert 'streamlit' in content
    os.remove(dashboard_path)  # Cleanup

def test_generate_contributor_graph(visualizer):
    html = visualizer.generate_contributor_graph()
    assert '<svg' in html
    assert 'd3' in html
    assert 'simulation' in html

def test_export_dashboard(visualizer):
    # Placeholder test: check no error
    visualizer.export_dashboard('html')

# Add more: heatmap, etc.