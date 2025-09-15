/**
 * Creates a text sprite for a graph node.
 * @param {Object} node - The node object.
 * @returns {THREE.Sprite} - The created text sprite.
 */
function createNodeLabel(node) {
  const text = node.label;
  if (!text) return null;

  // Render text at a higher resolution for clarity, then scale down
  const scale = 8;
  const fontSize = (node.isTitle ? 3 : 2) * scale;
  const font = `Bold ${fontSize}px Arial`;

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  context.font = font;

  let lines = [text];
  let textWidth = context.measureText(text).width;
  const lineHeight = fontSize * 1.2;

  if (text.length > 20) {
    const middle = Math.round(text.length / 2);
    let breakPoint = text.lastIndexOf(" ", middle);
    if (breakPoint === -1) breakPoint = middle;

    const line1 = text.substring(0, breakPoint);
    const line2 = text.substring(breakPoint + 1);
    lines = [line1, line2];

    textWidth = Math.max(
      context.measureText(line1).width,
      context.measureText(line2).width
    );
  }

  canvas.width = textWidth;
  canvas.height = lineHeight * lines.length;

  context.font = font;
  context.fillStyle = "rgba(255, 255, 255, 0.95)";
  context.textAlign = "center";
  context.textBaseline = "middle";

  lines.forEach((line, index) => {
    const y = (canvas.height / lines.length) * (index + 0.5);
    context.fillText(line, textWidth / 2, y);
  });

  const texture = new THREE.CanvasTexture(canvas);
  const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
  const sprite = new THREE.Sprite(spriteMaterial);

  sprite.scale.set(canvas.width / scale, canvas.height / scale, 1.0);
  const yOffset = node.isTitle ? -10 : -8;
  sprite.position.set(0, yOffset, 0);

  return sprite;
}

/**
 * Initializes and configures the 3D force graph.
 * @param {HTMLElement} container - The DOM element to contain the graph.
 * @param {{nodes: Array<Object>, links: Array<Object>}} graphData - The data for the graph.
 */
export function initializeGraph(container, { nodes, links }) {
  const myGraph = ForceGraph3D();

  myGraph(container)
    .graphData({ nodes, links })
    .nodeAutoColorBy("author")
    .onNodeClick((node) => {
      const distance = 40;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
      myGraph.cameraPosition(
        {
          x: node.x * distRatio,
          y: node.y * distRatio,
          z: node.z * distRatio,
        },
        node,
        3000
      );
    })
    .nodeThreeObject((node) => {
      const group = new THREE.Group();

      const geometry = new THREE.SphereGeometry(node.isTitle ? 5 : 3);
      const material = new THREE.MeshLambertMaterial({
        color: node.isTitle
          ? "purple"
          : node.author === "user"
          ? "lightblue"
          : "lightgreen",
        transparent: false,
        opacity: 1,
      });
      const sphere = new THREE.Mesh(geometry, material);
      group.add(sphere);

      const labelSprite = createNodeLabel(node);
      if (labelSprite) {
        group.add(labelSprite);
      }

      return group;
    });

  myGraph.d3Force("link").distance(() => 20);
  myGraph.d3Force("charge").strength(-200);

  const scene = myGraph.scene();
  const ambientLight = new THREE.AmbientLight(0xbbbbbb);
  scene.add(ambientLight);
}
