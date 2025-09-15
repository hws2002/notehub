const graphContainer = document.getElementById("graph-container");

fetch("../data/mock_data.json")
  .then((res) => res.json())
  .then((data) => {
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

    const myGraph = ForceGraph3D();
    myGraph(graphContainer)
      .graphData({ nodes, links })
      .nodeLabel("label")
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

        const geometry = new THREE.SphereGeometry(node.isTitle ? 4 : 2);
        const material = new THREE.MeshLambertMaterial({
          color: node.isTitle
            ? "purple"
            : node.author === "user"
            ? "lightblue"
            : "lightgreen",
          transparent: true,
          opacity: 0.8,
        });
        const sphere = new THREE.Mesh(geometry, material);
        group.add(sphere);

        const text = node.label;
        if (text) {
          const canvas = document.createElement("canvas");
          const context = canvas.getContext("2d");
          context.font = `Bold ${node.isTitle ? "14" : "10"}px Arial`;
          const textWidth = context.measureText(text).width;
          canvas.width = textWidth;
          canvas.height = node.isTitle ? 16 : 12;

          context.font = `Bold ${node.isTitle ? "14" : "10"}px Arial`;
          context.fillStyle = "rgba(255, 255, 255, 0.95)";
          context.fillText(text, 0, node.isTitle ? 14 : 10);

          const texture = new THREE.CanvasTexture(canvas);
          const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
          const sprite = new THREE.Sprite(spriteMaterial);

          sprite.scale.set(canvas.width / 2, canvas.height / 2, 1.0);
          sprite.position.set(node.isTitle ? 6 : 4, 0, 0);

          group.add(sprite);
        }

        return group;
      });

    myGraph.d3Force("link").distance((link) => 30);
    myGraph.d3Force("charge").strength(-120);

    const scene = myGraph.scene();
    const ambientLight = new THREE.AmbientLight(0xbbbbbb);
    scene.add(ambientLight);
  });
