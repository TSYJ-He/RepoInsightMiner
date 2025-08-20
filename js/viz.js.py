// D3.js visualization scripts for RepoInsightMiner
// This file contains reusable D3 functions for graphs like contributor dynamics, bug heatmaps.

// Force-directed graph for contributor dynamics
function renderForceGraph(container, data, width = 800, height = 600) {
    const svg = d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-100))
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
        .attr("r", 8)
        .attr("fill", d => color(d.group))
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text(d => d.id);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Heatmap for bug predictions (example)
function renderBugHeatmap(container, data, width = 800, height = 600) {
    // data: array of {file: str, risk: float}
    const svg = d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateReds)
        .domain([0, 1]);

    const barWidth = width / data.length;
    svg.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d, i) => i * barWidth)
        .attr("y", 0)
        .attr("width", barWidth - 1)
        .attr("height", height)
        .attr("fill", d => colorScale(d.risk));

    svg.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .attr("x", (d, i) => i * barWidth + barWidth / 2)
        .attr("y", height - 10)
        .attr("text-anchor", "middle")
        .text(d => d.file.slice(0, 10) + '...');  // Truncate labels
}

// Export for use in HTML or Streamlit embeds
export { renderForceGraph, renderBugHeatmap };