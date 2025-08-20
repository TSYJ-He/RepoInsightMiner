import argparse
import logging
import os
import json
from src.history_parser import HistoryParser
from src.graph_builder import GraphBuilder
from src.miner import Miner
from src.visualizer import Visualizer
from src.simulator import Simulator
from src.utils import timestamp_str, save_json

logger = logging.getLogger(__name__)

def main():
    """
    CLI entry point for RepoInsightMiner.
    Usage: repoinsightminer-cli --repo-url <url> [--github-token <token>] [--max-commits <int>] [--simulate <scenario> <params_json>]
    Outputs insights to JSON, generates viz HTML.
    Robust: Argument validation, error handling.
    Generalizable: Extendable commands.
    Convenient: Defaults, JSON outputs.
    Elegant: Modular parsing.
    """
    parser = argparse.ArgumentParser(description="AI-powered GitHub repo analyzer CLI.")
    parser.add_argument('--repo-url', required=True, help="GitHub repo URL (e.g., https://github.com/user/repo)")
    parser.add_argument('--github-token', default=os.getenv('GITHUB_TOKEN'), help="GitHub API token (or set GITHUB_TOKEN env)")
    parser.add_argument('--max-commits', type=int, default=1000, help="Max commits to parse")
    parser.add_argument('--no-prs', action='store_true', help="Skip parsing PRs")
    parser.add_argument('--output-dir', default='data/output', help="Directory for output files")
    parser.add_argument('--simulate', nargs=2, metavar=('SCENARIO', 'PARAMS_JSON'), help="Run simulation: scenario params_json")
    args = parser.parse_args()

    if not args.github_token:
        raise ValueError("GitHub token required. Set --github-token or GITHUB_TOKEN env.")

    os.environ['GITHUB_TOKEN'] = args.github_token
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Parse history
        hp = HistoryParser(args.repo_url)
        history = hp.parse_history(max_commits=args.max_commits, include_prs=not args.no_prs)

        # Build graph
        gb = GraphBuilder(history)
        nx_g = gb.get_networkx_graph()
        pyg_d = gb.to_pytorch_geometric()

        # Mine insights
        miner = Miner(nx_g, pyg_d)
        insights = miner.mine_insights(history)

        # Output insights
        insights_path = os.path.join(args.output_dir, f"insights_{timestamp_str()}.json")
        save_json(insights, insights_path)
        logger.info(f"Insights saved to {insights_path}")

        # Visualize
        viz = Visualizer(nx_g, insights)
        dashboard_path = os.path.join(args.output_dir, f"dashboard_{timestamp_str()}.py")
        viz.generate_dashboard(dashboard_path)
        logger.info(f"Run dashboard: streamlit run {dashboard_path}")

        # Optional simulation
        if args.simulate:
            scenario, params_json = args.simulate
            params = json.loads(params_json)
            sim = Simulator(nx_g, pyg_d, miner)
            sim_result = sim.run_simulation(scenario, params)
            sim_path = os.path.join(args.output_dir, f"simulation_{timestamp_str()}.json")
            save_json(sim_result, sim_path)
            logger.info(f"Simulation result saved to {sim_path}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()