import logging
import os
from typing import Dict, Any, List, Tuple
import networkx as nx
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.data import Data
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from src.utils import load_json, save_json

logger = logging.getLogger(__name__)

# Load spaCy model (assume downloaded)
nlp = spacy.load("en_core_web_sm")

# VADER for sentiment
sentiment_analyzer = SentimentIntensityAnalyzer()

class Miner:
    """
    AI-driven miner for insights from the knowledge graph.
    Features: Bug prediction (GNN), architecture smells (graph metrics + pylint),
    contributor dynamics (sentiment on PR comments), anomaly detection.
    Robust: Handles incomplete graphs, generalizable: Customizable models/queries,
    convenient: Cached results, elegant: Modular insight generators.
    """
    def __init__(self, nx_graph: nx.DiGraph, pyg_data: Data = None, cache_dir: str = 'data/cache', model_path: str = 'models/pretrained/bug_model.pt'):
        """
        Initialize with NetworkX graph and optional PyG data.
        :param nx_graph: Built from GraphBuilder.
        :param pyg_data: PyG data for GNN (if None, convert internally).
        :param cache_dir: For caching insights.
        :param model_path: Path to pretrained GNN model (placeholder for training).
        """
        self.G = nx_graph
        self.pyg_data = pyg_data or self._convert_to_pyg()
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.insights_cache_path = os.path.join(cache_dir, f"{self._get_repo_name()}_insights.json")
        os.makedirs(cache_dir, exist_ok=True)
        self.model = self._load_or_init_model()

    def _get_repo_name(self) -> str:
        """Infer repo name from graph nodes (assume metadata node or fallback)."""
        return next((n for n in self.G if self.G.nodes[n].get('type') == 'metadata'), 'unknown').replace('/', '_')

    def _convert_to_pyg(self) -> Data:
        """Placeholder conversion if not provided (simplified, assume from GraphBuilder)."""
        # In practice, use GraphBuilder.to_pytorch_geometric()
        node_list = list(self.G.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}
        edge_index = torch.tensor([(node_index[u], node_index[v]) for u, v in self.G.edges()]).t().contiguous()
        x = torch.eye(len(node_list))  # Identity features as placeholder
        return Data(x=x, edge_index=edge_index)

    def _load_or_init_model(self):
        """Load pretrained GNN or initialize a simple GraphSAGE for bug prediction."""
        class BugPredictor(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.conv2(x, edge_index)
                return torch.sigmoid(x)  # Probability of bug-prone

        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            model = BugPredictor(self.pyg_data.num_node_features, 16, 1)
            model.load_state_dict(torch.load(self.model_path))
        else:
            logger.warning("No pretrained model found. Initializing new (train before use).")
            model = BugPredictor(self.pyg_data.num_node_features, 16, 1)
        return model.eval()  # Set to eval mode

    def mine_insights(self, history_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mine all insights: bugs, smells, contributors."""
        if os.path.exists(self.insights_cache_path):
            logger.info(f"Loading cached insights from {self.insights_cache_path}")
            return load_json(self.insights_cache_path)

        insights = {
            'bug_predictions': self.predict_bugs(),
            'architecture_smells': self.detect_smells(),
            'contributor_dynamics': self.analyze_contributors(history_data),
        }
        save_json(insights, self.insights_cache_path)
        logger.info(f"Saved insights to {self.insights_cache_path}")
        return insights

    def predict_bugs(self) -> List[Dict[str, float]]:
        """Use GNN to predict bug-prone files (node classification)."""
        with torch.no_grad():
            preds = self.model(self.pyg_data.x, self.pyg_data.edge_index).squeeze().tolist()
        file_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == 'file']
        predictions = []
        for i, node in enumerate(file_nodes):  # Assume order matches pyg nodes
            risk = preds[i] if i < len(preds) else 0.0
            predictions.append({'file': node, 'bug_risk': risk, 'recommendation': 'Refactor' if risk > 0.7 else 'Monitor'})
        return sorted(predictions, key=lambda x: x['bug_risk'], reverse=True)

    def detect_smells(self) -> Dict[str, Any]:
        """Detect architecture smells: God classes (high centrality), code smells via pylint."""
        smells = {
            'god_classes': self._find_god_classes(),
            'code_smells': self._run_pylint_on_files(),  # Placeholder: needs actual file content
        }
        return smells

    def _find_god_classes(self) -> List[str]:
        """High degree centrality nodes (files with many changes/connections)."""
        centrality = nx.degree_centrality(self.G)
        file_nodes = {n: c for n, c in centrality.items() if self.G.nodes[n]['type'] == 'file'}
        threshold = 0.1  # Adjustable
        return [node for node, cent in sorted(file_nodes.items(), key=lambda x: x[1], reverse=True) if cent > threshold]

    def _run_pylint_on_files(self) -> List[Dict[str, Any]]:
        """Run pylint on file contents (assume files downloaded separately; placeholder)."""
        # For real use, need to clone repo and run on files
        reporter = CollectingReporter()
        # Example: Run('path/to/file.py', reporter=reporter)
        # For demo, return mock
        return [{'file': 'example.py', 'smells': ['too-many-lines', 'god-class']}]

    def analyze_contributors(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sentiment analysis on PR comments, ownership graph."""
        if not history_data or 'prs' not in history_data:
            logger.warning("No PR data provided for contributor analysis.")
            return {}

        sentiments = []
        for pr in history_data['prs']:
            for comment in pr['comments']:
                score = sentiment_analyzer.polarity_scores(comment['body'])
                comment['sentiment'] = score['compound']
                sentiments.append(comment)

        # Ownership: Cluster files by author contributions
        ownership = {}
        for node, data in self.G.nodes(data=True):
            if data['type'] == 'author':
                owned_files = [v for u, v in self.G.edges(node) if self.G.nodes[v]['type'] == 'file']  # Via commits
                ownership[node] = list(set(owned_files))  # Unique

        return {
            'average_sentiment': sum(c['sentiment'] for c in sentiments) / len(sentiments) if sentiments else 0,
            'sentiments': sentiments,
            'ownership': ownership,
        }

    def train_bug_model(self, labeled_data: Data, epochs: int = 10, save: bool = True):
        """Train the GNN model on labeled data (e.g., bug labels)."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(labeled_data.x, labeled_data.edge_index).squeeze()
            loss = criterion(out[labeled_data.train_mask], labeled_data.y[labeled_data.train_mask].float())
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch+1}: Loss {loss.item()}")

        if save:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Trained model saved to {self.model_path}")

# Example usage (for testing)
if __name__ == "__main__":
    # Assume graph from builder
    from src.graph_builder import GraphBuilder
    sample_history = load_json('data/cache/octocat_Hello-World_history.json')
    if sample_history:
        builder = GraphBuilder(sample_history)
        nx_g = builder.get_networkx_graph()
        pyg_d = builder.to_pytorch_geometric()
        miner = Miner(nx_g, pyg_d)
        insights = miner.mine_insights(sample_history)
        print(insights)