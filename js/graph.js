const categoryColors = {
  "AI & Finance": "#FF6347", // Tomato
  "Language & Linguistics": "#4682B4", // SteelBlue
  "Language & Healthcare": "#32CD32", // LimeGreen
  "Language & Translation": "#FFD700", // Gold
  "Machine Learning & DevOps": "#6A5ACD", // SlateBlue
  "Consumer Electronics": "#FF4500", // OrangeRed
  "Web Development": "#1E90FF", // DodgerBlue
  "Machine Learning & NLP": "#9932CC", // DarkOrchid
  "Data Science": "#20B2AA", // LightSeaGreen
  Programming: "#8A2BE2", // BlueViolet
  "Machine Learning": "#C71585", // MediumVioletRed
  Mathematics: "#5F9EA0", // CadetBlue
  "Statistics & Data Science": "#DAA520", // GoldenRod
  DevOps: "#D2691E", // Chocolate
};

const defaultColor = "#FFFFFF"; // White

/**
 * Returns a color for a node based on its category.
 * @param {Object} node - The node object, which may have a 'category' property.
 * @returns {string} The hex color code for the node.
 */
function getNodeColor(node) {
  return categoryColors[node.category] || defaultColor;
}

/**
 * Creates a text sprite for a graph node.
 * @param {Object} node - The node object, containing 'id' and 'label'.
 * @param {string} node.id - The unique identifier of the node.
 * @param {string} node.label - The display text for the node.
 * @returns {THREE.Sprite} - The created text sprite.
 */
function createNodeLabel(node) {
  const text = node.label;
  if (!text) return null;

  const scale = 12;
  const fontSize = 3 * scale;
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

  const yOffset = -10;
  sprite.position.set(0, yOffset, 0);

  return sprite;
}

/**
 * Creates a starfield background for the scene.
 * @returns {THREE.Points} - The starfield points object.
 */
function createStarfield() {
  const starQty = 5000;
  const starVertices = [];
  for (let i = 0; i < starQty; i++) {
    const x = (Math.random() - 0.5) * 2000;
    const y = (Math.random() - 0.5) * 2000;
    const z = (Math.random() - 0.5) * 2000;
    starVertices.push(x, y, z);
  }

  const starsGeometry = new THREE.BufferGeometry();
  starsGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(starVertices, 3)
  );

  const starsMaterial = new THREE.PointsMaterial({ color: 0x888888 });
  const starField = new THREE.Points(starsGeometry, starsMaterial);
  return starField;
}

/**
 * Initializes and configures the 3D force graph.
 * @param {HTMLElement} container - The DOM element to contain the graph.
 * @param {Object} graphData - The data for the graph.
 * @param {Array<Object>} graphData.nodes - An array of node objects.
 * @param {Array<Object>} graphData.links - An array of link objects.
 */
export function initializeGraph(container, { nodes, links }) {
  const myGraph = ForceGraph3D();

  myGraph(container)
    .graphData({ nodes, links })
    .linkWidth(0.5)
    .linkDirectionalParticles(1)
    .linkColor(() => "rgba(255, 255, 255, 0.6)")
    .backgroundColor("#000003")
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

      const baseSize = 4;
      const maxSize = 5; // Node radius will not exceed this value
      const size = Math.min(maxSize, baseSize + (node.degree || 0) * 0.5);
      const geometry = new THREE.SphereGeometry(size);
      const material = new THREE.MeshBasicMaterial({
        color: getNodeColor(node),
        transparent: false,
        opacity: 0.9,
      });
      const sphere = new THREE.Mesh(geometry, material);
      group.add(sphere);

      const labelSprite = createNodeLabel(node);
      if (labelSprite) {
        group.add(labelSprite);
      }

      return group;
    });

  myGraph.d3Force("link").distance(() => 40);
  myGraph.d3Force("charge").strength(-150);

  const scene = myGraph.scene();
  scene.add(createStarfield());

  const ambientLight = new THREE.AmbientLight(0xbbbbbb, 0.8);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);
}
