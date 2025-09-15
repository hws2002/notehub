import * as THREE from "https://cdn.skypack.dev/three@0.132.2";
import { GraphNode } from "./GraphNode.js";

export function createGraph(scene) {
  const nodes = [];
  const edges = [];

  // Shared geometry and material for efficiency
  const sphereGeometry = new THREE.SphereGeometry(0.8, 32, 32);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.5,
  });

  // Create initial spheres
  nodes.push(
    new GraphNode(
      1,
      "Sphere 1",
      new THREE.Vector3(0, 0, 0),
      0xffffff,
      sphereGeometry,
      scene
    )
  );
  nodes.push(
    new GraphNode(
      2,
      "Sphere 2",
      new THREE.Vector3(2, 1, -1),
      0xff0000,
      sphereGeometry,
      scene
    )
  );
  nodes.push(
    new GraphNode(
      3,
      "Sphere 3",
      new THREE.Vector3(-2, -1, 1),
      0x0000ff,
      sphereGeometry,
      scene
    )
  );

  // Add 20 more random spheres
  for (let i = 4; i <= 23; i++) {
    const position = new THREE.Vector3(
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20
    );
    const color = new THREE.Color(Math.random() * 0xffffff);
    const node = new GraphNode(
      i,
      `Node ${i}`,
      position,
      color,
      sphereGeometry,
      scene
    );
    nodes.push(node);
  }

  // --- Create random edges (lines) between nodes ---
  const numberOfEdges = 30;
  const existingEdges = new Set();

  for (let i = 0; i < numberOfEdges; i++) {
    let nodeA = nodes[Math.floor(Math.random() * nodes.length)];
    let nodeB = nodes[Math.floor(Math.random() * nodes.length)];

    const edgeKey1 = `${nodeA.id}-${nodeB.id}`;
    const edgeKey2 = `${nodeB.id}-${nodeA.id}`;
    if (nodeA.id === nodeB.id || existingEdges.has(edgeKey1)) {
      i--;
      continue;
    }

    const points = [nodeA.sphere.position, nodeB.sphere.position];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, lineMaterial);
    scene.add(line);

    const edge = { source: nodeA, target: nodeB, line: line };
    edges.push(edge);

    nodeA.connectedEdges.push(edge);
    nodeB.connectedEdges.push(edge);
    existingEdges.add(edgeKey1);
    existingEdges.add(edgeKey2);
  }

  return { nodes, edges };
}
