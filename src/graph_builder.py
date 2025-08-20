import os
import logging
import json
from typing import Dict, Any, List, Tuple
import networkx as nx
from torch_geometric.data import Data
import torch
from src.utils import load_json, save_json, timestamp_str

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Builds a knowledge graph from parsed GitHub history data.
    Nodes: Files, Authors, Commits.
    Edges: Contributions, Changes, Dependencies (basic).
    Supports conversion to PyTorch Geometric for GNN usage.
    Designed for robustness (handles missing data), generalization (any repo structure),
    convenience (caching), and elegance (modular node/edge addition).
    """
    def __init__(self, history_data: Dict[str, Any], cache_dir: str = 'data/cache'):
        """
        Initialize with parsed history data.
        :param history_data: Output from HistoryParser.parse_history().
        :param cache_dir: Directory for caching graph files.
        """
        self.history_data = history_data
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.graph_cache_path = os.path.join(cache_dir, f"{self._get_repo_name()}_graph.gpickle")
        self.pyg_cache_path = os.path.join(cache_dir, f"{self._get_repo_name()}_pyg.pt")
        self.G: nx.DiGraph = self._load_or_build_graph()

    def _get_repo_name(self) -> str:
        """Extract repo name from metadata."""
        return self.history_data.get('metadata', {}).get('name', 'unknown').replace('/', '_')

    def _load_or_build_graph(self) -> nx.DiGraph:
        """Load graph from cache if exists, else build."""
        if os.path.exists(self.graph_cache_path):
            logger.info(f"Loading cached graph from {self.graph_cache_path}")
            return nx.read_gpickle(self.graph_cache_path)
        logger.info("Building new graph...")
        G = nx.DiGraph()
        self._add_nodes(G)
        self._add_edges(G)
        self._save_graph(G)
        return G

    def _add_nodes(self, G: nx.DiGraph):
        """Add nodes for authors, commits, files with attributes."""
        # Authors
        authors = set()
        for commit in self.history_data.get('commits', []):
            authors.add(commit['author'])
        for author in authors:
            G.add_node(author, type='author', commits=0)  # Count updated later

        # Commits
        for commit in self.history_data.get('commits', []):
            G.add_node(commit['sha'], type='commit', date=commit['date'], message=commit['message'])

        # Files (track unique files across commits)
        files = set()
        for commit in self.history_data.get('commits', []):
            for file in commit['files']:
                files.add(file['filename'])
        for file in files:
            G.add_node(file, type='file', changes=0, bug_prone=False)  # Placeholders

        logger.info(f"Added nodes: {G.number_of_nodes()} (Authors: {len(authors)}, Commits: {len(self.history_data.get('commits', []))}, Files: {len(files)})")

    def _add_edges(self, G: nx.DiGraph):
        """Add edges: Author -> Commit, Commit -> File (with change type)."""
        for commit in self.history_data.get('commits', []):
            author = commit['author']
            sha = commit['sha']
            if author in G and sha in G:
                G.add_edge(author, sha, type='authored', weight=1)
                G.nodes[author]['commits'] += 1  # Update count

            for file in commit['files']:
                filename = file['filename']
                if sha in G and filename in G:
                    G.add_edge(sha, filename, type=file['status'], additions=file['additions'],
                               deletions=file['deletions'], changes=file['changes'])

                    G.nodes[filename]['changes'] += file['changes']  # Update change count

        # Basic file-file dependencies (placeholder: infer from filenames/extensions)
        files = [n for n, d in G.nodes(data=True) if d['type'] == 'file']
        for i, file1 in enumerate(files):
            for file2 in files[i+1:]:
                if self._infer_dependency(file1, file2):
                    G.add_edge(file1, file2, type='depends_on', weight=1)

        logger.info(f"Added edges: {G.number_of_edges()}")

    def _infer_dependency(self, file1: str, file2: str) -> bool:
        """Simple heuristic for file dependencies (e.g., .py importing others). Extendable."""
        # Placeholder: True if same directory or common extensions
        dir1 = os.path.dirname(file1)
        dir2 = os.path.dirname(file2)
        return dir1 == dir2 and file1 != file2  # Basic co-location

    def _save_graph(self, G: nx.DiGraph):
        """Save NetworkX graph to file."""
        nx.write_gpickle(G, self.graph_cache_path)
        logger.info(f"Saved graph to {self.graph_cache_path}")

    def get_networkx_graph(self) -> nx.DiGraph:
        """Return the NetworkX graph."""
        return self.G

    def to_pytorch_geometric(self, node_features: Dict[str, torch.Tensor] = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data.
        :param node_features: Optional dict of node features (key: node, value: tensor).
        :return: PyG Data object.
        """
        if os.path.exists(self.pyg_cache_path):
            logger.info(f"Loading cached PyG data from {self.pyg_cache_path}")
            return torch.load(self.pyg_cache_path)

        # Map nodes to indices
        node_list = list(self.G.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}

        # Edge index
        edge_index_list: List[Tuple[int, int]] = []
        for u, v in self.G.edges():
            edge_index_list.append((node_index[u], node_index[v]))
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # Node features (basic: one-hot type + attributes)
        num_nodes = len(node_list)
        x = torch.zeros((num_nodes, 3), dtype=torch.float)  # 3 features: type (author/commit/file)
        for node, idx in node_index.items():
            node_data = self.G.nodes[node]
            node_type = node_data['type']
            if node_type == 'author':
                x[idx, 0] = 1.0
                x[idx, 2] = node_data.get('commits', 0) / 100.0  # Normalized
            elif node_type == 'commit':
                x[idx, 1] = 1.0
            elif node_type == 'file':
                x[idx, 2] = 1.0
                x[idx, 1] = node_data.get('changes', 0) / 1000.0  # Normalized

        if node_features:
            # Merge custom features
            custom_x = torch.stack([node_features.get(node, torch.zeros(1)) for node in node_list])
            x = torch.cat([x, custom_x], dim=1)

        data = Data(x=x, edge_index=edge_index)
        torch.save(data, self.pyg_cache_path)
        logger.info(f"Saved PyG data to {self.pyg_cache_path}")
        return data

    def add_custom_edges(self, edges: List[Tuple[str, str, Dict[str, Any]]]):
        """Convenience method to add custom edges for extension."""
        for u, v, attrs in edges:
            if u in self.G and v in self.G:
                self.G.add_edge(u, v, **attrs)
        self._save_graph(self.G)

# Example usage (for testing)
if __name__ == "__main__":
    # Assume history_data loaded from parser
    sample_history = load_json('data/cache/octocat_Hello-World_history.json')  # Replace with actual path
    if sample_history:
        builder = GraphBuilder(sample_history)
        G = builder.get_networkx_graph()
        print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        pyg_data = builder.to_pytorch_geometric()
        print(f"PyG data: {pyg_data}")