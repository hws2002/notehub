import { getGraphData } from "./dataProcessor.js";
import { initializeGraph } from "./graph.js";

const graphContainer = document.getElementById("graph-container");
const dataUrl = "../data/mock_data.json";

async function main() {
  try {
    const { nodes, links } = await getGraphData(dataUrl);
    initializeGraph(graphContainer, { nodes, links });
  } catch (error) {
    console.error("Failed to initialize graph:", error);
    graphContainer.textContent = "Failed to load graph data.";
  }
}

main();
