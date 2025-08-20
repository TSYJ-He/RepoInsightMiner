import os
import logging
import json
from typing import Dict, Any, List
import streamlit as st
import networkx as nx
from networkx.readwrite import json_graph
from src.utils import save_json, timestamp_str

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Generates interactive visualizations and dashboards for repo insights.
    Features: Contributor graphs, bug heatmaps, architecture diagrams using D3.js embedded in Streamlit.
    Robust: Handles large graphs (subsampling), generalizable: Custom viz types,
    convenient: Export to HTML/PDF, elegant: Modular viz generators with templates.
    """
    def __init__(self, nx_graph: nx.DiGraph, insights: Dict[str, Any], output_dir: str = 'data/viz'):
        """
        Initialize with graph and mined insights.
        :param nx_graph: From GraphBuilder.
        :param insights: From Miner.mine_insights().
        :param output_dir: Directory for saving viz files.
        """
        self.G = nx_graph
        self.insights = insights
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_dashboard(self, dashboard_path: str = None):
        """
        Run Streamlit dashboard interactively.
        If dashboard_path provided, save as script; else run inline.
        """
        if dashboard_path:
            self._write_streamlit_script(dashboard_path)
            logger.info(f"Streamlit dashboard script saved to {dashboard_path}. Run with 'streamlit run {dashboard_path}'")
        else:
            self._run_streamlit_dashboard()

    def _run_streamlit_dashboard(self):
        """Inline Streamlit app for quick viz (non-blocking)."""
        st.title("RepoInsightMiner Dashboard")

        st.header("Contributor Dynamics")
        contributor_html = self.generate_contributor_graph()
        st.components.v1.html(contributor_html, height=600)

        st.header("Bug Predictions")
        bugs = self.insights.get('bug_predictions', [])
        st.dataframe(bugs)

        st.header("Architecture Smells")
        smells = self.insights.get('architecture_smells', {})
        st.json(smells)

        # Add more sections as needed

    def _write_streamlit_script(self, path: str):
        """Generate a standalone Streamlit script."""
        script_content = f"""
import streamlit as st
import json

# Load data (assume JSON files; in practice, load from cache)
with open('data/cache/insights.json', 'r') as f:  # Replace with actual path
    insights = json.load(f)

st.title("RepoInsightMiner Dashboard")

st.header("Contributor Dynamics")
# Embed D3 graph (generate or load HTML)
contributor_html = \"\"\"{self.generate_contributor_graph()}\"\"\"
st.components.v1.html(contributor_html, height=600)

st.header("Bug Predictions")
st.dataframe(insights.get('bug_predictions', []))

st.header("Architecture Smells")
st.json(insights.get('architecture_smells', {{}}))
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script_content)

    def generate_contributor_graph(self) -> str:
        """Generate D3.js force-directed graph HTML for contributors and files."""
        graph_data = self._prepare_d3_data()
        html_template = self._get_d3_template()
        html = html_template.format(json_data=json.dumps(graph_data))
        output_path = os.path.join(self.output_dir, f"contributor_graph_{timestamp_str()}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Saved contributor graph to {output_path}")
        return html

    def _prepare_d3_data(self) -> Dict[str, Any]:
        """Convert subset of NX graph to D3 JSON format (nodes, links). Subsample for large graphs."""
        subgraph = self.G.subgraph(list(self.G.nodes())[:500])  # Subsample if too large
        data = json_graph.node_link_data(subgraph)
        # Enhance with types/colors
        for node in data['nodes']:
            node_type = subgraph.nodes[node['id']]['type']
            node['group'] = 1 if node_type == 'author' else 2 if node_type == 'commit' else 3
        return data

    def _get_d3_template(self) -> str:
        """D3.js force-directed graph template."""
        return """
<!DOCTYPE html>
<meta charset="utf-8">
<body>
<svg width="960" height="600"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {json_data};

const svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

const color = d3.scaleOrdinal(d3.schemeCategory10);

const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id))
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(width / 2, height / 2));

const link = svg.append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
  .selectAll("line")
  .data(data.links)
  .join("line")
    .attr("stroke-width", d => Math.sqrt(d.value || 1));

const node = svg.append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
  .selectAll("circle")
  .data(data.nodes)
  .join("circle")
    .attr("r", 5)
    .attr("fill", d => color(d.group))
    .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

node.append("title")
    .text(d => d.id);

simulation.on("tick", () => {{
  link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

  node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
}});

function dragstarted(event, d) {{
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}}

function dragged(event, d) {{
  d.fx = event.x;
  d.fy = event.y;
}}

function dragended(event, d) {{
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}}
</script>
"""

    def generate_bug_heatmap(self) -> str:
        """Generate D3 heatmap for bug risks (placeholder)."""
        # Similar to above, but for heatmap
        return "<div>Heatmap placeholder</div>"  # Extend with actual D3 code

    def export_dashboard(self, format: str = 'html'):
        """Export full dashboard to file."""
        # Use streamlit's export or generate static HTML
        logger.info(f"Exporting to {format} (placeholder implementation)")

# Example usage (for testing)
if __name__ == "__main__":
    # Assume graph and insights loaded
    from src.graph_builder import GraphBuilder
    from src.miner import Miner
    sample_history = {'metadata': {'name': 'test'}, 'commits': []}  # Mock
    builder = GraphBuilder(sample_history)
    nx_g = builder.get_networkx_graph()
    miner = Miner(nx_g)
    insights = miner.mine_insights()
    viz = Visualizer(nx_g, insights)
    viz.generate_dashboard()