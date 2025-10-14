import { initializeGraph } from "./graph.js";

const graphContainer = document.getElementById("graph-container");
const dataUrl1 = "../graph_data/graph_data.json";
const dataUrl2 = "../graph_data/graph_data_llm.json";

async function main() {
  try {
    const [res1, res2] = await Promise.all([
      fetch(`${dataUrl1}?t=${new Date().getTime()}`),
      fetch(`${dataUrl2}?t=${new Date().getTime()}`),
    ]);

    const data1 = await res1.json();
    const data2 = await res2.json();

    const combinedNodes = [...data1.nodes];
    const nodeIds = new Set(data1.nodes.map((n) => n.id));

    for (const node of data2.nodes) {
      if (!nodeIds.has(node.id)) {
        combinedNodes.push(node);
        nodeIds.add(node.id);
      }
    }

    const combinedLinks = [...data1.links, ...data2.links];

    // Calculate the degree (number of links) for each node
    const degree = {};
    combinedLinks.forEach((link) => {
      degree[link.source] = (degree[link.source] || 0) + 1;
      degree[link.target] = (degree[link.target] || 0) + 1;
    });

    combinedNodes.forEach((node) => {
      node.degree = degree[node.id] || 0;
    });

    initializeGraph(graphContainer, {
      nodes: combinedNodes,
      links: combinedLinks,
    });
  } catch (error) {
    console.error("Failed to initialize graph:", error);
    graphContainer.textContent = "Failed to load graph data.";
  }
}

main();
