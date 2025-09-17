/**
 * Creates a text sprite for a graph node.
 * @param {Object} node - The node object.
 * @returns {THREE.Sprite} - The created text sprite.
 */
function createNodeLabel(node) {
  const text = node.label;
  if (!text) return null;

  // 렌더링 매개변수: 폰트 해상도 및 크기
  // scale 값을 높이면 폰트가 더 선명해집니다 (고해상도로 렌더링 후 축소).
  const scale = 12;
  const fontSize = (node.isTitle ? 3 : 2) * scale; // isTitle은 제목 노드, 나머지는 일반 노드 크기
  const font = `Bold ${fontSize}px Arial`;

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  context.font = font;

  let lines = [text];
  let textWidth = context.measureText(text).width;
  const lineHeight = fontSize * 1.2;

  // 긴 텍스트를 두 줄로 나눔
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
  // yOffset: 노드 구체(sphere)로부터 라벨의 수직 위치 조정
  const yOffset = node.isTitle ? -10 : -8;
  sprite.position.set(0, yOffset, 0);

  return sprite;
}

/**
 * Creates a starfield background for the scene.
 * @returns {THREE.Points} - The starfield points object.
 */
function createStarfield() {
  const starQty = 10000; // 렌더링 매개변수: 별 배경의 별 개수
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
 * @param {{nodes: Array<Object>, links: Array<Object>}} graphData - The data for the graph.
 */
export function initializeGraph(container, { nodes, links }) {
  const myGraph = ForceGraph3D();

  myGraph(container)
    .graphData({ nodes, links })
    // 렌더링 매개변수: 링크(선) 속성
    .linkWidth(0.5) // 링크의 두께
    .linkDirectionalParticles(1) // 링크를 따라 움직이는 입자 수. 0으로 설정하면 보이지 않음.
    .linkColor(() => "rgba(255, 255, 255, 0.6)") // 링크 색상 및 투명도
    .backgroundColor("#000003")
    // 노드 클릭 시 카메라 이동 애니메이션
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

      // 렌더링 매개변수: 노드 크기
      // 노드 크기를 연결된 링크(degree) 수에 따라 동적으로 결정합니다.
      // 최소 크기를 2로 설정하고, 연결이 많을수록 커집니다.
      const size = 2 + (node.degree || 0) * 0.5;
      const geometry = new THREE.SphereGeometry(size);
      const material = new THREE.MeshBasicMaterial({
        color: "purple", // All nodes are currently the same type, so we use one color.
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

  // 렌더링 매개변수: 물리 엔진 힘(Force) 조정
  // 'link' force: 노드 간의 간격(거리)을 설정합니다. 값이 클수록 멀어집니다.
  myGraph.d3Force("link").distance(() => 40);
  // 'charge' force: 노드들이 서로 밀어내는 힘. 음수 값이 클수록(예: -200) 더 강하게 밀어냅니다.
  myGraph.d3Force("charge").strength(-150);

  // 렌더링 매개변수: 조명(Luminosity) 및 배경
  const scene = myGraph.scene();
  scene.add(createStarfield());

  // AmbientLight: 씬 전체에 고르게 비추는 조명. (색상, 강도)
  const ambientLight = new THREE.AmbientLight(0xbbbbbb, 0.8); // 강도를 높이면 전체적으로 밝아짐
  scene.add(ambientLight);
  // DirectionalLight: 특정 방향에서 오는 조명 (태양광과 유사). (색상, 강도)
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8); // 강도를 높이면 하이라이트가 더 밝아짐
  directionalLight.position.set(1, 1, 1); // coming from top-right-front
  scene.add(directionalLight);
}
