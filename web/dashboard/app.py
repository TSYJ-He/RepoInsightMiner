import streamlit as st
import json
import os
from src.history_parser import HistoryParser
from src.graph_builder import GraphBuilder
from src.miner import Miner
from src.visualizer import Visualizer
from src.simulator import Simulator
from src.utils import get_github_token

# Streamlit Dashboard App
st.set_page_config(page_title="RepoInsightMiner Dashboard", layout="wide")


@st.cache_data
def analyze_repo(repo_url: str, max_commits: int = 1000, include_prs: bool = True):
    """Cached analysis function."""
    parser = HistoryParser(repo_url)
    history = parser.parse_history(max_commits, include_prs)

    builder = GraphBuilder(history)
    nx_g = builder.get_networkx_graph()
    pyg_d = builder.to_pytorch_geometric()

    miner = Miner(nx_g, pyg_d)
    insights = miner.mine_insights(history)

    return history, nx_g, pyg_d, insights, miner


def main():
    st.title("RepoInsightMiner: AI-Powered GitHub Repo Analyzer")

    # Sidebar for input
    with st.sidebar:
        st.header("Analysis Settings")
        repo_url = st.text_input("GitHub Repo URL", value="https://github.com/octocat/Hello-World")
        max_commits = st.number_input("Max Commits to Parse", min_value=10, max_value=5000, value=1000)
        include_prs = st.checkbox("Include Pull Requests", value=True)
        analyze_button = st.button("Analyze Repo")

        st.header("Simulation")
        scenario = st.selectbox("Scenario", ["remove_contributor", "add_changes"])
        params = st.text_area("Params (JSON)", value='{"contributor": "octocat"}')
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=1)
        simulate_button = st.button("Run Simulation")

    if analyze_button:
        with st.spinner("Analyzing repo..."):
            history, nx_g, pyg_d, insights, miner = analyze_repo(repo_url, max_commits, include_prs)
            st.session_state['nx_g'] = nx_g
            st.session_state['pyg_d'] = pyg_d
            st.session_state['insights'] = insights
            st.session_state['miner'] = miner
            st.session_state['history'] = history

    if 'insights' in st.session_state:
        insights = st.session_state['insights']
        nx_g = st.session_state['nx_g']

        st.header("Repo Metadata")
        st.json(st.session_state['history']['metadata'])

        st.header("Bug Predictions")
        st.dataframe(insights['bug_predictions'])

        st.header("Architecture Smells")
        st.json(insights['architecture_smells'])

        st.header("Contributor Dynamics")
        st.json(insights['contributor_dynamics'])

        st.header("Interactive Graph Visualization")
        viz = Visualizer(nx_g, insights)
        contributor_html = viz.generate_contributor_graph()
        st.components.v1.html(contributor_html, height=600, width=1200)

    if simulate_button and 'nx_g' in st.session_state:
        try:
            params_dict = json.loads(params)
            simulator = Simulator(st.session_state['nx_g'], st.session_state['pyg_d'], st.session_state['miner'])
            with st.spinner("Running simulation..."):
                sim_result = simulator.run_simulation(scenario, params_dict, num_runs)
            st.header("Simulation Results")
            st.json(sim_result)
        except json.JSONDecodeError:
            st.error("Invalid JSON in params.")


if __name__ == "__main__":
    main()