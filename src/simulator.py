import copy
import logging
import os
from typing import Dict, Any, List
import networkx as nx
import torch
from torch_geometric.data import Data
from src.miner import Miner  # For re-mining after simulation
from src.utils import save_json, timestamp_str

logger = logging.getLogger(__name__)

class Simulator:
    """
    Runs hypothetical "what-if" scenarios on the knowledge graph.
    Features: Simulate removing/adding contributors, changes, predict impacts (e.g., bug increase).
    Robust: Deep copies graph to avoid mutation, generalizable: Custom scenarios,
    convenient: Scenario templates, elegant: Chainable simulations with diff reporting.
    """
    def __init__(self, nx_graph: nx.DiGraph, pyg_data: Data, miner: Miner, cache_dir: str = 'data/cache'):
        """
        Initialize with original graph, PyG data, and Miner instance.
        :param nx_graph: Original NetworkX graph.
        :param pyg_data: Original PyG data.
        :param miner: Miner instance for insight re-computation.
        :param cache_dir: For saving simulation results.
        """
        self.original_G = nx_graph
        self.original_pyg = pyg_data
        self.miner = miner
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.sim_cache_path = os.path.join(cache_dir, f"{self._get_repo_name()}_sim_results.json")

    def _get_repo_name(self) -> str:
        """Infer repo name."""
        return next((n for n in self.original_G if self.original_G.nodes[n].get('type') == 'metadata'), 'unknown').replace('/', '_')

    def run_simulation(self, scenario: str, params: Dict[str, Any], num_runs: int = 1) -> Dict[str, Any]:
        """
        Run a simulation scenario.
        :param scenario: Type e.g., 'remove_contributor', 'add_changes'.
        :param params: Scenario-specific params e.g., {'contributor': 'user'}).
        :param num_runs: For Monte Carlo (average over runs).
        :return: Dict with original_insights, simulated_insights, diff.
        """
        results = []
        for _ in range(num_runs):
            sim_G = copy.deepcopy(self.original_G)
            sim_pyg = self._copy_pyg_data(self.original_pyg)
            self._apply_scenario(sim_G, sim_pyg, scenario, params)
            sim_miner = Miner(sim_G, sim_pyg, model_path=self.miner.model_path)  # New miner for sim graph
            sim_insights = sim_miner.mine_insights()
            results.append(sim_insights)

        # Average results if multiple runs (placeholder: simple avg for numerics)
        avg_insights = self._average_insights(results) if num_runs > 1 else results[0]

        original_insights = self.miner.mine_insights()
        diff = self._compute_diff(original_insights, avg_insights)

        sim_result = {
            'scenario': scenario,
            'params': params,
            'original': original_insights,
            'simulated': avg_insights,
            'diff': diff,
        }
        save_json(sim_result, self.sim_cache_path)
        logger.info(f"Saved simulation result to {self.sim_cache_path}")
        return sim_result

    def _copy_pyg_data(self, data: Data) -> Data:
        """Deep copy PyG data."""
        return Data(x=data.x.clone(), edge_index=data.edge_index.clone())

    def _apply_scenario(self, G: nx.DiGraph, pyg: Data, scenario: str, params: Dict[str, Any]):
        """Apply modifications based on scenario."""
        if scenario == 'remove_contributor':
            contributor = params.get('contributor')
            if contributor in G:
                # Remove node and edges
                G.remove_node(contributor)
                # Update PyG: Remove node and adjust indices (simplified; in practice, rebuild PyG)
                self._rebuild_pyg_from_nx(G, pyg)
            else:
                logger.warning(f"Contributor {contributor} not found.")

        elif scenario == 'add_changes':
            file = params.get('file')
            changes = params.get('changes', 10)
            if file in G:
                G.nodes[file]['changes'] += changes
                # Update features in PyG
                node_idx = list(G.nodes()).index(file)  # Assume order
                pyg.x[node_idx, 1] += changes / 1000.0  # Normalized

        # Add more scenarios as needed

    def _rebuild_pyg_from_nx(self, G: nx.DiGraph, pyg: Data):
        """Rebuild PyG data after graph changes (efficient update)."""
        # Placeholder: Full rebuild for simplicity
        node_list = list(G.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}
        edge_index_list = [(node_index[u], node_index[v]) for u, v in G.edges()]
        pyg.edge_index = torch.tensor(edge_index_list).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
        # Features: Recompute similar to GraphBuilder
        num_nodes = len(node_list)
        pyg.x = torch.zeros((num_nodes, pyg.x.size(1)), dtype=torch.float)
        # ... Reassign features ...

    def _average_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average numerical insights over runs (e.g., bug risks)."""
        if not results:
            return {}
        avg = copy.deepcopy(results[0])
        for key in ['bug_predictions']:
            if key in avg:
                for i in range(len(avg[key])):
                    risks = [r[key][i]['bug_risk'] for r in results]
                    avg[key][i]['bug_risk'] = sum(risks) / len(risks)
        # Extend for other numerics
        return avg

    def _compute_diff(self, original: Dict[str, Any], simulated: Dict[str, Any]) -> Dict[str, Any]:
        """Compute differences between original and simulated insights."""
        diff = {}
        for key in original:
            if key in simulated:
                if isinstance(original[key], list) and all(isinstance(item, dict) for item in original[key]):
                    # e.g., bug_predictions
                    diff[key] = [{'file': o['file'], 'risk_diff': s['bug_risk'] - o['bug_risk']}
                                 for o, s in zip(original[key], simulated[key]) if o['file'] == s['file']]
                elif isinstance(original[key], dict):
                    # Recursive diff
                    diff[key] = self._compute_diff(original[key], simulated[key])
                else:
                    diff[key] = simulated[key] - original[key] if isinstance(original[key], (int, float)) else 'changed'
        return diff

    def get_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Return example scenario templates for convenience."""
        return {
            'remove_contributor': {'params': {'contributor': 'username'}, 'description': 'Simulate impact of removing a contributor.'},
            'add_changes': {'params': {'file': 'path/to/file', 'changes': 50}, 'description': 'Simulate additional changes to a file.'},
        }

# Example usage (for testing)
if __name__ == "__main__":
    # Assume from previous modules
    from src.graph_builder import GraphBuilder
    from src.miner import Miner
    sample_history = {'metadata': {'name': 'test'}, 'commits': []}  # Mock
    builder = GraphBuilder(sample_history)
    nx_g = builder.get_networkx_graph()
    pyg_d = builder.to_pytorch_geometric()
    miner = Miner(nx_g, pyg_d)
    sim = Simulator(nx_g, pyg_d, miner)
    result = sim.run_simulation('remove_contributor', {'contributor': 'some_author'})
    print(result)