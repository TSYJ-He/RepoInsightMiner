import pytest
import os
import torch
from src.miner import Miner
from src.graph_builder import GraphBuilder
from src.utils import load_json

@pytest.fixture
def sample_history():
    # Reuse or mock
    sample_path = 'tests/sample_history.json'
    if os.path.exists(sample_path):
        return load_json(sample_path)
    else:
        return {
            'metadata': {'name': 'test_repo'},
            'commits': [
                {'sha': 'abc123', 'author': 'user1', 'date': '2023-01-01', 'message': 'Commit',
                 'files': [{'filename': 'file1.py', 'status': 'added', 'additions': 10, 'deletions': 0, 'changes': 10, 'diff': ''}]}
            ],
            'prs': [
                {'number': 1, 'title': 'PR1', 'author': 'user1', 'created_at': '2023-01-02', 'merged_at': None,
                 'comments': [{'author': 'user2', 'body': 'Looks good!', 'created_at': '2023-01-03', 'sentiment': None}]}
            ]
        }

@pytest.fixture
def builder(sample_history):
    return GraphBuilder(sample_history, cache_dir='tests/cache')

@pytest.fixture
def miner(builder):
    nx_g = builder.get_networkx_graph()
    pyg_d = builder.to_pytorch_geometric()
    return Miner(nx_g, pyg_d, model_path='tests/mock_model.pt')

def test_mine_insights(miner, sample_history):
    insights = miner.mine_insights(sample_history)
    assert 'bug_predictions' in insights
    assert 'architecture_smells' in insights
    assert 'contributor_dynamics' in insights
    assert len(insights['bug_predictions']) > 0
    assert 'god_classes' in insights['architecture_smells']

def test_predict_bugs(miner):
    preds = miner.predict_bugs()
    assert isinstance(preds, list)
    assert all('file' in p and 'bug_risk' in p for p in preds)

def test_analyze_contributors(miner, sample_history):
    dynamics = miner.analyze_contributors(sample_history)
    assert 'average_sentiment' in dynamics
    assert 'sentiments' in dynamics
    assert 'ownership' in dynamics
    assert dynamics['average_sentiment'] != 0  # Assuming positive comment

def test_train_bug_model(miner):
    # Mock labeled data
    labeled_data = miner.pyg_data.clone()
    labeled_data.train_mask = torch.ones(labeled_data.num_nodes, dtype=torch.bool)
    labeled_data.y = torch.rand(labeled_data.num_nodes)
    miner.train_bug_model(labeled_data, epochs=1, save=False)
    # No assert, just check no error

# Add more: smells, etc.