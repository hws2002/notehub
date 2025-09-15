const stopWords = new Set([
  "the",
  "a",
  "an",
  "in",
  "is",
  "it",
  "of",
  "for",
  "on",
  "with",
  "to",
  "and",
  "what",
  "how",
  "why",
  "i",
]);

/**
 * Processes raw conversation data into nodes and links for the graph.
 * @param {Array<Object>} data - The raw conversation data from the JSON file.
 * @returns {{nodes: Array<Object>, links: Array<Object>}} - The processed nodes and links.
 */
function processData(data) {
  let nodes = [];
  let links = [];

  data.forEach((conversation) => {
    const conversationId = conversation.title.replace(/\s+/g, "_");
    nodes.push({
      id: conversationId,
      label: conversation.title,
      isTitle: true,
    });

    Object.values(conversation.mapping).forEach((message) => {
      nodes.push({
        id: message.id,
        label: message.message.content.parts[0],
        author: message.message.author.role,
      });

      if (message.parent) {
        links.push({
          source: message.parent,
          target: message.id,
        });
      } else {
        links.push({
          source: conversationId,
          target: message.id,
        });
      }
    });
  });

  return { nodes, links };
}

/**
 * Creates additional links between nodes based on shared significant words.
 * @param {Array<Object>} nodes - The array of graph nodes.
 * @param {Array<Object>} links - The array of existing graph links.
 * @returns {Array<Object>} - The updated array of links.
 */
function createWordBasedLinks(nodes, links) {
  const messageNodes = nodes.filter((n) => !n.isTitle && n.label);
  const wordMap = new Map();

  // 1. Map words to the nodes that contain them
  messageNodes.forEach((node) => {
    const words = node.label.toLowerCase().match(/\b(\w+)\b/g) || [];
    words.forEach((word) => {
      if (word.length > 3 && !stopWords.has(word)) {
        if (!wordMap.has(word)) {
          wordMap.set(word, []);
        }
        wordMap.get(word).push(node.id);
      }
    });
  });

  // 2. Create links between nodes that share words
  const existingLinks = new Set(links.map((l) => `${l.source}>${l.target}`));
  wordMap.forEach((nodeIds) => {
    if (nodeIds.length > 1 && nodeIds.length < 10) {
      for (let i = 0; i < nodeIds.length; i++) {
        for (let j = i + 1; j < nodeIds.length; j++) {
          const source = nodeIds[i];
          const target = nodeIds[j];
          const linkId1 = `${source}>${target}`;
          const linkId2 = `${target}>${source}`;
          if (!existingLinks.has(linkId1) && !existingLinks.has(linkId2)) {
            links.push({ source, target, isGenerated: true });
            existingLinks.add(linkId1);
          }
        }
      }
    }
  });

  return links;
}

/**
 * Fetches and processes graph data from the specified URL.
 * @param {string} url - The URL of the data file.
 * @returns {Promise<{nodes: Array<Object>, links: Array<Object>}>}
 */
export async function getGraphData(url) {
  const res = await fetch(url);
  const data = await res.json();

  let { nodes, links } = processData(data);
  links = createWordBasedLinks(nodes, links);

  return { nodes, links };
}
