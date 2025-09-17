import { initializeGraph } from "./graph.js";

const graphContainer = document.getElementById("graph-container");
const dataUrl = "../data/graph_data.json";

async function main() {
  try {
    const res = await fetch(dataUrl);
    const { nodes, links } = await res.json();

    // Calculate the degree (number of links) for each node
    const degree = {};
    links.forEach((link) => {
      degree[link.source] = (degree[link.source] || 0) + 1;
      degree[link.target] = (degree[link.target] || 0) + 1;
    });

    nodes.forEach((node) => {
      node.degree = degree[node.id] || 0;
    });

    initializeGraph(graphContainer, { nodes, links });
  } catch (error) {
    console.error("Failed to initialize graph:", error);
    graphContainer.textContent = "Failed to load graph data.";
  }
}

main();
