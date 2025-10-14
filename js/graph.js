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

// 카테고리가 지정되지 않은 노드의 기본 색상입니다.
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

  // 라벨의 크기를 조절하는 변수입니다. 값을 키우면 라벨이 커집니다.
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

  // 노드 중심으로부터 라벨의 Y축 위치를 조절합니다. 음수 값은 라벨을 위로 올립니다.
  const yOffset = -10;
  sprite.position.set(0, yOffset, 0);

  return sprite;
}

/**
 * Creates a starfield background for the scene.
 * @returns {THREE.Points} - The starfield points object.
 */
function createStarfield() {
  // 배경에 표시될 별의 개수입니다.
  const starQty = 5000;
  const starVertices = [];
  for (let i = 0; i < starQty; i++) {
    // 별들이 생성될 공간의 크기를 조절합니다. 현재는 가로/세로/깊이 2000의 정육면체 공간입니다.
    const range = 4000;
    const x = (Math.random() - 0.5) * range;
    const y = (Math.random() - 0.5) * range;
    const z = (Math.random() - 0.5) * range;
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
    // 링크(선)의 두께를 조절합니다.
    .linkWidth(0.5)
    // 링크를 따라 움직이는 파티클의 개수입니다. 0으로 설정하면 보이지 않습니다.
    .linkDirectionalParticles(1)
    .linkColor(() => "rgba(255, 255, 255, 0.6)")
    .backgroundColor("#000003")
    .onNodeClick((node) => {
      // 노드 클릭 시 카메라가 얼마나 가까이 다가갈지 결정합니다. 값이 클수록 멀어집니다.
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

      // 노드의 기본 크기입니다.
      const baseSize = 5;
      const maxSize = 6;
      // 연결된 링크 수(node.degree)에 따라 노드 크기를 동적으로 조절합니다.
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

  // --- D3 Force-Directed Layout Configuration ---

  // 1. forceLink: 링크(선)의 기본 길이를 설정합니다. 값이 클수록 노드들이 서로 멀리 떨어집니다.
  myGraph.d3Force("link").distance(100); // 링크 길이를 100으로 설정

  // 2. forceManyBody (charge): 노드 간의 인력/척력을 조절합니다. 음수 값이 클수록 노드들이 서로를 강하게 밀어냅니다.
  myGraph.d3Force("charge").strength(-120); // -180의 힘으로 서로 밀어내도록 설정

  // 3. forceCenter: 그래프의 모든 노드를 중앙(0,0,0)으로 끌어당겨 흩어지는 것을 방지합니다.
  // strength를 높여(기본값 0.1) 노드들을 중앙으로 더 강하게 끌어당깁니다.
  myGraph.d3Force("center", d3.forceCenter().strength(2));

  // 4. forceCollide: 노드들이 서로 겹치지 않도록 충돌을 감지하고 밀어냅니다. 반지름 값은 노드 크기와 비슷하게 설정하는 것이 좋습니다.
  // 이 값을 크게 유지하여 노드들이 서로 충분한 거리를 유지하도록 합니다.
  myGraph.d3Force("collide", d3.forceCollide(30)); // 충돌 반경을 30으로 설정
  const scene = myGraph.scene();
  scene.add(createStarfield());

  const ambientLight = new THREE.AmbientLight(0xbbbbbb, 0.8);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);
}
