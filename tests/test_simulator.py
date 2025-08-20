import pytest
import copy
import os
from src.simulator import Simulator
from src.graph_builder import GraphBuilder
from src.miner import Miner
from src.utils import load_json

@pytest.fixture
def sample_history():
    # Mock or load
    return {
        'metadata': {'name': 'test_repo'},
        'commits': [
            {'sha': 'abc123', 'author': 'user1', 'date': '2023-01-01', 'message': 'Commit',
             'files': [{'filename': 'file1.py', 'status': 'added', 'additions': 10, 'deletions': 0, 'changes': 10, 'diff': ''}]}
        ]
    }

@pytest.fixture
def builder(sample_history):
    return GraphBuilder(sample_history)

@pytest.fixture
def nx_g(builder):
    return builder.get_networkx_graph()

@pytest.fixture
def pyg_d(builder):
    return builder.to_pytorch_geometric()

@pytest.fixture
def miner(nx_g, pyg_d):
    return Miner(nx_g, pyg_d)

@pytest.fixture
def simulator(nx_g, pyg_d, miner):
    return Simulator(nx_g, pyg_d, miner, cache_dir='tests/cache')

def test_run_simulation(simulator):
    scenario = 'remove_contributor'
    params = {'contributor': 'user1'}
    result = simulator.run_simulation(scenario, params)
    assert 'scenario' in result
    assert result['scenario'] == scenario
    assert 'original' in result
    assert 'simulated' in result
    assert 'diff' in result
    # Check graph was modified
    assert 'user1' not in result['simulated']['contributor_dynamics'].get('ownership', {})

def test_apply_scenario(simulator):
    sim_G = copy.deepcopy(simulator.original_G)
    sim_pyg = simulator._copy_pyg_data(simulator.original_pyg)
    scenario = 'add_changes'
    params = {'file': 'file1.py', 'changes': 20}
    simulator._apply_scenario(sim_G, sim_pyg, scenario, params)
    assert sim_G.nodes['file1.py']['changes'] == 30  # Original 10 + 20

def test_compute_diff(simulator):
    original = {'bug_predictions': [{'file': 'file1.py', 'bug_risk': 0.5}]}
    simulated = {'bug_predictions': [{'file': 'file1.py', 'bug_risk': 0.7}]}
    diff = simulator._compute_diff(original, simulated)
    assert diff['bug_predictions'][0]['risk_diff'] == 0.2

def test_get_scenario_templates(simulator):
    templates = simulator.get_scenario_templates()
    assert 'remove_contributor' in templates
    assert 'description' in templates['remove_contributor']

# Add more: multiple runs, edge cases